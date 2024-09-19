import os
import torch
import wandb
import torch.nn as nn
import numpy as np
from resnet import resnet18
from utils import train_epoch, test_epoch, get_data_loaders, replace_and_test
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def transfer_linear_probe(model):
    """
    Transfer learning on CIFAR-10 using a linear probe.
    """
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
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
    optimizer = torch.optim.Adam(model.fc, lr=learning_rate)

    os.makedirs('./ckpts', exist_ok=True)

    best_test_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        train_epoch(epoch, model, train_loader, optimizer, criterion, device, None)

        test_loss, _ = test_epoch(epoch, model, test_loader, criterion, device)

        # save every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'./ckpts/resnet18_cifar100_to_cifar10_epoch{epoch}.pth')

        # Save the model with the best test loss
        if test_loss < best_test_loss:
            print(f'Find new best model at Epoch {epoch}')
            best_test_loss = test_loss
            torch.save(model.state_dict(), './ckpts/resnet18_cifar100_to_cifar10_best.pth')

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
    train_features, train_labels = extract_features(feature_extractor, cifar10_train_loader)
    test_features, test_labels = extract_features(feature_extractor, cifar10_test_loader)

    # Train the k-NN classifier
    knn.fit(train_features, train_labels)

    # Test the k-NN classifier
    predictions = knn.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    print(f'Accuracy of k-NN classifier: {accuracy:.2f}')
    return knn


def main():
    # Get the model pre-trained on CIFAR-100
    mode = 'normal'
    ckpt_folder = os.path.join('./ckpts', mode)
    model = resnet18().to(device)
    model.load_state_dict(torch.load(os.path.join(ckpt_folder, f'resnet18_cifar100_epoch200.pth')))

    # Transfer learning on CIFAR-10 using a linear probe
    beta_vals = np.arange(0.8, 1, 0.01)
    _, test_loader = get_data_loaders('cifar10', 128)
    model = transfer_linear_probe(model)
    dataset = 'cifar100_to_cifar10'
    replace_and_test(model, test_loader, beta_vals, mode, dataset)


if __name__ == '__main__':
    wandb.init(project='smooth-spline', entity='leyang_hu')
    main()
    wandb.finish()
