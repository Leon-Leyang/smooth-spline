import torch


def eval_multi_label(model, testloader, device='cuda'):
    """
    Computes mean accuracy (mA) for multi-label classification.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in testloader:
            if isinstance(targets, list) and isinstance(targets[0], torch.Tensor):  # Hack for handling celeb_a
                targets = torch.stack(targets, dim=0)  # shape: (40, batch_size)
                targets = targets.T
                targets = targets.float()  # ensure float if needed

            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu()

            # Convert ground-truth from -1/+1 to 0/1:
            targets_01 = (targets + 1) // 2

            all_preds.append(probs)
            all_targets.append(targets_01)

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)  # shape: (N, 40)
    all_targets = torch.cat(all_targets, dim=0)  # shape: (N, 40)

    # Threshold at 0.5 to get binary predictions in {0,1}
    pred_labels = (all_preds >= 0.5).float()

    # Per-attribute accuracy => mean across attributes
    correct_by_attr = (pred_labels == all_targets).float().sum(dim=0)  # shape: (40,)
    total_samples = all_targets.size(0)
    acc_by_attr = correct_by_attr / total_samples  # shape: (40,)

    # Mean accuracy across attributes.
    mA = acc_by_attr.mean().item()

    # Compute per-attribute F1 scores.
    # True Positives, False Positives, and False Negatives per attribute.
    TP = ((pred_labels == 1) & (all_targets == 1)).float().sum(dim=0)
    FP = ((pred_labels == 1) & (all_targets == 0)).float().sum(dim=0)
    FN = ((pred_labels == 0) & (all_targets == 1)).float().sum(dim=0)

    # Compute F1 score: F1 = 2TP / (2TP + FP + FN).
    # A small epsilon (1e-8) is added to the denominator to avoid division by zero.
    f1_by_attr = 2 * TP / (2 * TP + FP + FN + 1e-8)

    return mA * 100.0, acc_by_attr * 100.0, f1_by_attr * 100.0


def eval_mse(model, testloader, device='cuda'):
    """
    Computes Mean Squared Error (MSE) for regression tasks.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(device)
            targets = targets.to(device).float()  # Ensure targets are floats
            preds = model(inputs).cpu()  # Get model predictions

            all_preds.append(preds)
            all_targets.append(targets.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)  # shape: (N,)
    all_targets = torch.cat(all_targets, dim=0)  # shape: (N,)

    # Compute Mean Squared Error (MSE)
    mse = torch.mean((all_preds - all_targets) ** 2).item()
    return mse
