import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
import copy
import matplotlib.pyplot as plt
from resnet import resnet18
from utils import WarmUpLR, ReplacementMapping, replace_module, get_file_name, train_epoch, test_epoch, get_data_loaders
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(mode):
    """
    Train the model on CIFAR-100.
    :param mode: mode of the train, e.g. normal/overfit
    """
    assert mode in ['normal', 'overfit'], 'Mode must be either normal or overfit'

    # Hyperparameters
    batch_size = 128
    learning_rate = 0.1
    num_epochs = 200

    # Get the data loaders for CIFAR-100
    cifar100_train_loader, cifar100_test_loader = get_data_loaders('cifar100', batch_size)

    # Initialize the model
    model = resnet18()
    model = model.to(device)

    # Return the model if it has already been trained
    ckpt_folder = os.path.join('./ckpts', mode)
    os.makedirs(f'{ckpt_folder}', exist_ok=True)
    file_to_check = os.path.join(ckpt_folder, f'resnet18_cifar100_epoch{num_epochs}.pth')
    if os.path.exists(file_to_check):
        model.load_state_dict(torch.load(file_to_check))
        print(f'Loaded model from {file_to_check}')
        return model, cifar100_test_loader

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # Learning rate scheduler with specific milestones for reduction
    if mode == 'normal':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    else:
        scheduler = None

    # Warmup scheduler
    if mode == 'normal':
        iter_per_epoch = len(cifar100_train_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch)
    else:
        warmup_scheduler = None

    best_test_loss = float('inf')

    # Train the model
    for epoch in range(1, num_epochs + 1):
        if epoch > 1:
            if scheduler is not None:
                scheduler.step(epoch)

        train_epoch(epoch, model, cifar100_train_loader, optimizer, criterion, device, warmup_scheduler)
        test_loss, _ = test_epoch(epoch, model, cifar100_test_loader, criterion, device)

        # save every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_folder, f'resnet18_cifar100_epoch{epoch}.pth'))

        # Save the model with the best test loss
        if test_loss < best_test_loss:
            print(f'Find new best model at Epoch {epoch}')
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(ckpt_folder, 'resnet18_cifar100_best.pth'))

    return model, cifar100_test_loader


def transfer_linear_probe(model):
    """
    Transfer learning on CIFAR-10 using a linear probe.
    """
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 50

    # Get the data loaders for CIFAR-10
    cifar10_train_loader, cifar10_test_loader = get_data_loaders('cifar10', batch_size)

    # Replace the last layer of the model with a linear layer for CIFAR-10
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.fc.weight.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc, lr=learning_rate)

    os.makedirs('./ckpts', exist_ok=True)

    best_test_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        train_epoch(epoch, model, cifar10_train_loader, optimizer, criterion, device, None)

        test_loss, _ = test_epoch(epoch, model, cifar10_test_loader, criterion, device)

        # save every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'./ckpts/resnet18_cifar10_epoch{epoch}.pth')

        # Save the model with the best test loss
        if test_loss < best_test_loss:
            print(f'Find new best model at Epoch {epoch}')
            best_test_loss = test_loss
            torch.save(model.state_dict(), './ckpts/resnet18_cifar10_best.pth')

    return model


def extract_feautres(feature_extractor, dataloader):
    """
    Extract features from the model.
    """
    feature_extractor.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            features.append(feature_extractor(inputs).cpu().numpy())
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
    train_features, train_labels = extract_feautres(feature_extractor, cifar10_train_loader)
    test_features, test_labels = extract_feautres(feature_extractor, cifar10_test_loader)

    # Train the k-NN classifier
    knn.fit(train_features, train_labels)

    # Test the k-NN classifier
    predictions = knn.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    print(f'Accuracy of k-NN classifier: {accuracy:.2f}')
    return knn


def replace_and_test_cifar100(model, test_loader, beta_vals, mode):
    """
    Replace ReLU with BetaReLU and test the model on CIFAR-100.
    """
    assert mode in ['normal', 'overfit'], 'Mode must be either normal or overfit'

    print('*' * 50)
    print('Running post-replace experiment on CIFAR-100...')
    print('*' * 50)
    criterion = nn.CrossEntropyLoss()

    test_loss_list = []
    beta_list = []

    # Test the original model
    print('Testing the original model...')
    base_test_loss, _ = test_epoch(-1, model, test_loader, criterion, device)
    best_test_loss = base_test_loss
    best_beta = 1

    # Test the model with different beta values
    for i, beta in enumerate(beta_vals):
        print(f'Using BetaReLU with beta={beta:.5f}')
        replacement_mapping = ReplacementMapping(beta=beta)
        orig_model = copy.deepcopy(model)
        new_model = replace_module(orig_model, replacement_mapping)
        test_loss, _ = test_epoch(-1, new_model, test_loader, criterion, device)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_beta = beta
        test_loss_list.append(test_loss)
        beta_list.append(beta)
    test_loss_list.append(base_test_loss)
    beta_list.append(1)
    print(f'Best test loss: {best_test_loss:.6f} with beta={best_beta:.5f}, compared to ReLU test loss: {base_test_loss:.6f}')

    # Plot the test loss vs beta values
    plt.figure(figsize=(12, 6))
    plt.plot(beta_list, test_loss_list)
    plt.axhline(y=base_test_loss, color='r', linestyle='--', label='ReLU Test Loss')
    plt.xlabel('Beta')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs Beta Values')
    plt.gca().yaxis.get_major_formatter().set_useOffset(False)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.xticks(beta_list)
    plt.legend()
    output_folder = os.path.join("./figures", get_file_name(__file__))
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"replace_and_test_cifar100_{mode}.png"))
    plt.show()


def main():
    # Initialize Weights and Biases (wandb)
    wandb.init(project='smooth-spline', entity='leyang_hu')

    # Train the model on CIFAR-100
    mode = 'normal'
    model, cifar100_test_loader = train(mode)

    beta_vals = np.arange(0.9995, 1, 0.00001)

    # Replace ReLU with BetaReLU and test the model on CIFAR-100
    replace_and_test_cifar100(model, cifar100_test_loader, beta_vals, mode)

    wandb.finish()


if __name__ == '__main__':
    main()


