import os
import torch
import copy
import wandb
import torch.nn as nn
import torch.optim as optim
import numpy as np
from resnet import resnet18
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from utils import train_epoch, test_epoch, get_data_loaders, replace_and_test, ReplacementMapping, replace_module, get_file_name
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def transfer_linear_probe(model, mode, beta_val=None):
    """
    Transfer learning on CIFAR-10 using a linear probe.
    """
    assert mode in ['normal', 'suboptimal', 'overfit'], 'Mode must be either normal, suboptimal or overfit'

    # Hyperparameters
    batch_size = 128
    learning_rate = 0.005
    num_epochs = 50

    # Get the data loaders for CIFAR-10
    train_loader, test_loader = get_data_loaders('cifar10', batch_size)

    # Replace the last layer of the model with a linear layer for CIFAR-10
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.fc.weight.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.2)

    ckpt_folder = os.path.join('./ckpts', mode)
    os.makedirs(f'{ckpt_folder}', exist_ok=True)

    # Load the weights if the model has been trained before
    if not beta_val:
        file_to_check = os.path.join(ckpt_folder, f'resnet18_cifar100_to_cifar10_epoch{num_epochs}.pth')
    else:
        file_to_check = os.path.join(ckpt_folder, f'resnet18_cifar100_beta{beta_val:.2f}_to_cifar10_epoch{num_epochs}.pth')
    if os.path.exists(file_to_check):
        model.load_state_dict(torch.load(file_to_check, weights_only=True))
        print(f'Loaded model from {file_to_check}')
        return model

    best_test_loss = float('inf')

    wandb.init(project='smooth-spline', entity='leyang_hu')

    for epoch in range(1, num_epochs + 1):
        if epoch > 1:
            if scheduler is not None:
                scheduler.step(epoch)
        train_epoch(epoch, model, train_loader, optimizer, criterion, device, None)
        test_loss, _ = test_epoch(epoch, model, test_loader, criterion, device)

        # save every 10 epochs
        if epoch % 10 == 0:
            if not beta_val:
                torch.save(model.state_dict(), os.path.join(ckpt_folder, f'resnet18_cifar100_to_cifar10_epoch{epoch}.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(ckpt_folder, f'resnet18_cifar100_beta{beta_val:.2f}_to_cifar10_epoch{epoch}.pth'))

        # Save the model with the best test loss
        if test_loss < best_test_loss:
            print(f'Find new best model at Epoch {epoch}')
            best_test_loss = test_loss
            if not beta_val:
                torch.save(model.state_dict(), os.path.join(ckpt_folder, f'resnet18_cifar100_to_cifar10_best.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(ckpt_folder, f'resnet18_cifar100_beta{beta_val:.2f}_to_cifar10_best.pth'))

    wandb.finish()

    return model


def extract_features(feature_extractor, dataloader):
    """
    Extract features from the model.
    """
    feature_extractor.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            feature = feature_extractor(inputs)
            feature = torch.flatten(feature, 1)
            features.append(feature.cpu().numpy())
            labels.append(targets.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    return features, labels


def transfer_knn(model):
    """
    Transfer learning on CIFAR-10 using a k-NN classifier.
    """
    # Hyperparameters
    batch_size = 128
    neighbors = 5

    # Get the data loaders for CIFAR-10
    cifar10_train_loader, cifar10_test_loader = get_data_loaders('cifar10', batch_size)
    knn = KNeighborsClassifier(n_neighbors=neighbors)

    # Extract features from the pre-trained model
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    train_features, train_labels = extract_features(feature_extractor, cifar10_train_loader)
    test_features, test_labels = extract_features(feature_extractor, cifar10_test_loader)

    # Train the k-NN classifier
    knn.fit(train_features, train_labels)

    # Test the k-NN classifier
    predictions = knn.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    print(f'Accuracy of k-NN classifier: {accuracy:.4f}')
    return accuracy


def replace_and_test_linear_probe_on(mode, test_loader, beta_vals):
    ckpt_folder = os.path.join('./ckpts', mode)
    model = resnet18().to(device)
    model.load_state_dict(torch.load(os.path.join(ckpt_folder, f'resnet18_cifar100_epoch200.pth'), weights_only=True))
    model = transfer_linear_probe(model, mode)
    replace_and_test(model, test_loader, beta_vals, mode, 'cifar100_to_cifar10_linear', __file__)


def replace_and_test_knn_on(mode, beta_vals):
    ckpt_folder = os.path.join('./ckpts', mode)
    model = resnet18().to(device)
    model.load_state_dict(torch.load(os.path.join(ckpt_folder, f'resnet18_cifar100_epoch200.pth'), weights_only=True))

    acc_list = []
    beta_list = []

    # Test the original model
    print('Testing the original model')
    base_acc = transfer_knn(model)
    best_acc = base_acc
    best_beta = 1

    # Test the model with different beta values
    for i, beta in enumerate(beta_vals):
        print(f'Using BetaReLU with beta={beta:.6f}')
        replacement_mapping = ReplacementMapping(beta=beta)
        orig_model = copy.deepcopy(model)
        new_model = replace_module(orig_model, replacement_mapping)
        acc = transfer_knn(new_model)
        if acc > best_acc:
            best_acc = acc
            best_beta = beta
        acc_list.append(acc)
        beta_list.append(beta)
    acc_list.append(base_acc)
    beta_list.append(1)
    print(f'Best accuracy: {best_acc:.4f} with beta={best_beta:.6f}, compared to ReLU accuracy: {base_acc:.4f}')

    # Plot the test loss vs beta values
    plt.figure(figsize=(12, 8))
    plt.plot(beta_list, acc_list)
    plt.axhline(y=base_acc, color='r', linestyle='--', label='ReLU Test Accuracy')
    plt.xlabel('Beta')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs Beta Values')

    # Ensure that both x-axis and y-axis show raw numbers without offset or scientific notation
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.get_major_formatter().set_useOffset(False)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)

    plt.xticks(beta_list[::5], rotation=45)
    plt.legend()
    output_folder = os.path.join("./figures", get_file_name(__file__))
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"replace_and_test_cifar100_to_cifar10_knn_{mode}.png"))
    plt.show()


def main():
    # Transfer learning on CIFAR-10 using a linear probe and test the model with different beta values of BetaReLU
    _, test_loader = get_data_loaders('cifar10', 128)
    mode_2_beta_vals = {
        'normal': np.arange(0.95, 1, 0.001),
        'suboptimal': np.arange(0.95, 1, 0.001),
        'overfit': np.arange(0.95, 1, 0.001)
    }
    for mode, beta_vals in mode_2_beta_vals.items():
        replace_and_test_linear_probe_on(mode, test_loader, beta_vals)

    # Transfer learning on CIFAR-10 using a k-NN classifier and test the model with different beta values of BetaReLU
    mode_2_beta_vals = {
        'normal': np.arange(0.95, 1, 0.001),
        'suboptimal': np.arange(0.95, 1, 0.001),
        'overfit': np.arange(0.95, 1, 0.001)
    }
    for mode, beta_vals in mode_2_beta_vals.items():
        replace_and_test_knn_on(mode, beta_vals)


if __name__ == '__main__':
    main()
