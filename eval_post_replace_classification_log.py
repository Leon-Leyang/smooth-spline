import re
import os
from collections import defaultdict
import statistics


def extract_accuracies_from_file(file_path, robustness=False):
    new_accuracies = defaultdict(list)
    baseline_accuracies = defaultdict(list)
    best_betas = defaultdict(list)

    if robustness:
        # Regex for robust accuracy
        pattern = r"Best robust accuracy for ([\w\/\-\_]+) with ([\w]+) attack: ([0-9]+\.[0-9]+) with beta=([0-9\.]+), compared to ReLU accuracy: ([0-9]+\.[0-9]+)"
    else:
        # Regex for standard accuracy
        pattern = r"Best accuracy for ([\w\/\-\_]+): ([0-9]+\.[0-9]+) with beta=([0-9\.]+)"
        baseline_pattern = r"ReLU accuracy: ([0-9]+\.[0-9]+)"

    with open(file_path, 'r') as file:
        for line in file:
            if robustness:
                # Match robust accuracy lines
                match = re.search(pattern, line)
                if match:
                    ds_name = match.group(1)
                    attack_type = match.group(2)
                    dataset_with_attack = f"{ds_name}_{attack_type}"
                    accuracy = float(match.group(3))
                    beta = float(match.group(4))
                    baseline_accuracy = float(match.group(5))

                    new_accuracies[dataset_with_attack].append(accuracy)
                    best_betas[dataset_with_attack].append(beta)
                    baseline_accuracies[dataset_with_attack].append(baseline_accuracy)
            else:
                # Match standard accuracy lines
                match = re.search(pattern, line)
                baseline_match = re.search(baseline_pattern, line)

                if match:
                    current_dataset = match.group(1)
                    accuracy = float(match.group(2))
                    beta = float(match.group(3))
                    new_accuracies[current_dataset].append(accuracy)
                    best_betas[current_dataset].append(beta)

                    if baseline_match:
                        baseline_accuracy = float(baseline_match.group(1))
                        baseline_accuracies[current_dataset].append(baseline_accuracy)

    return new_accuracies, baseline_accuracies, best_betas


def compute_statistics(log_files, robustness=False):
    combined_new_accuracies = defaultdict(list)
    combined_baseline_accuracies = defaultdict(list)
    combined_betas = defaultdict(list)

    for log_file in log_files:
        file_new_accuracies, file_baseline_accuracies, file_betas = extract_accuracies_from_file(log_file, robustness)
        for dataset, accuracies in file_new_accuracies.items():
            combined_new_accuracies[dataset].extend(accuracies)
        for dataset, accuracies in file_baseline_accuracies.items():
            combined_baseline_accuracies[dataset].extend(accuracies)
        for dataset, betas in file_betas.items():
            combined_betas[dataset].extend(betas)

    # Compute mean and standard deviation for each dataset
    stats_new_accuracies = {
        dataset: (statistics.mean(values), statistics.stdev(values))
        for dataset, values in combined_new_accuracies.items()
    }
    stats_baseline_accuracies = {
        dataset: (statistics.mean(values), statistics.stdev(values))
        for dataset, values in combined_baseline_accuracies.items()
    }
    stats_betas = {
        dataset: (statistics.mean(values), statistics.stdev(values))
        for dataset, values in combined_betas.items()
    }

    # Compute normalized improvement for each dataset
    normalized_improvements = {
        dataset: ((stats_new_accuracies[dataset][0] - stats_baseline_accuracies[dataset][0]) /
                  stats_baseline_accuracies[dataset][0]) * 100
        if stats_baseline_accuracies[dataset][0] != 0 else 0
        for dataset in stats_new_accuracies
    }

    # Compute overall normalized improvement across all datasets
    total_normalized_improvement = sum(normalized_improvements.values()) / len(
        normalized_improvements) if normalized_improvements else 0

    return stats_new_accuracies, stats_baseline_accuracies, stats_betas, normalized_improvements, total_normalized_improvement


if __name__ == "__main__":
    # Input: List of log file paths
    log_files = [
        "./logs/post_replace_classification_lp_replace_coeff0.0_topk1_reg1_seed42.log",
        "./logs/post_replace_classification_lp_replace_coeff0.0_topk1_reg1_seed43.log",
        "./logs/post_replace_classification_lp_replace_coeff0.0_topk1_reg1_seed44.log"
    ]

    # Ensure files exist
    for file in log_files:
        if not os.path.exists(file):
            print(f"File not found: {file}")
            exit(1)

    # Compute statistics for standard accuracy
    stats_new_accuracies, stats_baseline_accuracies, stats_betas, normalized_improvements, total_normalized_improvement = compute_statistics(
        log_files, robustness=False)

    print("Standard Accuracy, Beta Statistics, and Normalized Improvements:")
    for dataset in stats_new_accuracies:
        new_mean, new_std = stats_new_accuracies[dataset]
        baseline_mean, baseline_std = stats_baseline_accuracies.get(dataset, (0, 0))
        beta_mean, beta_std = stats_betas.get(dataset, (0, 0))
        normalized_improvement = normalized_improvements.get(dataset, 0)
        print(f"Dataset: {dataset}, Beta: {beta_mean:.2f} ± {beta_std:.2f}, New Accuracy: {new_mean:.2f} ± {new_std:.2f}, Baseline Accuracy: {baseline_mean:.2f} ± {baseline_std:.2f}, Normalized Improvement: {normalized_improvement:.2f}%")

    print(f"Overall Normalized Improvement Across All Datasets: {total_normalized_improvement:.2f}%")

    # Compute statistics for robust accuracy
    stats_new_accuracies, stats_baseline_accuracies, stats_betas, normalized_improvements, total_normalized_improvement = compute_statistics(
        log_files, robustness=True)

    print("\n")
    print("Robust Accuracy, Beta Statistics, and Normalized Improvements:")
    for dataset in stats_new_accuracies:
        new_mean, new_std = stats_new_accuracies[dataset]
        baseline_mean, baseline_std = stats_baseline_accuracies.get(dataset, (0, 0))
        beta_mean, beta_std = stats_betas.get(dataset, (0, 0))
        normalized_improvement = normalized_improvements.get(dataset, 0)
        print(f"Dataset: {dataset}, Beta: {beta_mean:.2f} ± {beta_std:.2f}, New Accuracy: {new_mean:.2f} ± {new_std:.2f}, Baseline Accuracy: {baseline_mean:.2f} ± {baseline_std:.2f}, Normalized Improvement: {normalized_improvement:.2f}%")

    print(f"Overall Normalized Improvement Across All Datasets: {total_normalized_improvement:.2f}%")
