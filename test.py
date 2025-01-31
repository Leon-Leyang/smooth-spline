import statistics


def compute_mean_std(relu_accuracies, ct_accuracies, betas):
    """
    Computes mean and standard deviation for given lists of ReLU accuracies, CT accuracies, and betas.
    :param relu_accuracies: List of ReLU accuracy values
    :param ct_accuracies: List of CT accuracy values
    :param betas: List of beta values
    :return: Dictionary with computed statistics
    """
    stats = {}

    if relu_accuracies:
        stats['relu_accuracy_mean'] = statistics.mean(relu_accuracies)
        stats['relu_accuracy_std'] = statistics.stdev(relu_accuracies) if len(relu_accuracies) > 1 else 0
    else:
        stats['relu_accuracy_mean'] = 0
        stats['relu_accuracy_std'] = 0

    if ct_accuracies:
        stats['ct_accuracy_mean'] = statistics.mean(ct_accuracies)
        stats['ct_accuracy_std'] = statistics.stdev(ct_accuracies) if len(ct_accuracies) > 1 else 0
    else:
        stats['ct_accuracy_mean'] = 0
        stats['ct_accuracy_std'] = 0

    if betas:
        stats['beta_mean'] = statistics.mean(betas)
        stats['beta_std'] = statistics.stdev(betas) if len(betas) > 1 else 0
    else:
        stats['beta_mean'] = 0
        stats['beta_std'] = 0

    return stats


# Example usage
if __name__ == "__main__":
    relu_accuracy_values = [4.3857, 4.3959, 4.4097]
    ct_accuracy_values = [4.2883, 4.2643, 4.2933]
    beta_values = [0.57, 0.52, 0.51]

    results = compute_mean_std(relu_accuracy_values, ct_accuracy_values, beta_values)
    print(
        f"ReLU Accuracy Mean: {results['relu_accuracy_mean']:.4f}, ReLU Accuracy Std: {results['relu_accuracy_std']:.4f}")
    print(f"CT Accuracy Mean: {results['ct_accuracy_mean']:.4f}, CT Accuracy Std: {results['ct_accuracy_std']:.4f}")
    print(f"Beta Mean: {results['beta_mean']:.2f}, Beta Std: {results['beta_std']:.2f}")