import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
import numpy as np
import copy
from resnet import resnet18
from utils import WarmUpLR, ReplacementMapping, replace_module
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_data_loaders(dataset, batch_size=128):
    """
    Get the data loaders for the dataset
    """
    assert dataset in ['cifar10', 'cifar100']

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=8)
    return trainloader, testloader


def train_epoch(epoch, model, trainloader, optimizer, criterion, device, warmup_scheduler):
    """
    Train the model for one epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            train_loss = running_loss / (batch_idx + 1)
            train_accuracy = 100. * correct / total
            print(f'Epoch {epoch}, Step {batch_idx}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')

        if epoch <= 1:
            warmup_scheduler.step()

    # Log the training loss and accuracy to wandb
    wandb.log({'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_accuracy, 'lr': optimizer.param_groups[0]['lr']})


def test_epoch(epoch, model, testloader, criterion, device):
    """
    Test the model for one epoch
    Specify epoch=-1 to use for testing after training
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(testloader)
    test_accuracy = 100. * correct / total
    if epoch != -1:
        print(f'Test Epoch {epoch}, Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')

        # Log the test loss and accuracy to wandb
        wandb.log({'epoch': epoch, 'val_loss': test_loss, 'val_accuracy': test_accuracy})
    else:
        print(f'Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')

    return test_loss


def train():
    """
    Train the model on CIFAR-100
    """
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
    if os.path.exists(f'./ckpts/resnet18_cifar100_{num_epochs}.pth'):
        model.load_state_dict(torch.load(f'./ckpts/resnet18_cifar100_epoch{num_epochs}.pth'))
        print(f'Loaded model from ./ckpts/resnet18_cifar100_epoch{num_epochs}.pth')
        return model, cifar100_test_loader

    # Resume training if a checkpoint exists(resume from the latest epoch)
    # latest_epoch = 0
    # for epoch in range(1, num_epochs + 1):
    #     if os.path.exists(f'./ckpts/resnet18_cifar100_epoch{epoch}.pth'):
    #         latest_epoch = epoch
    # if latest_epoch > 0:
    #     model.load_state_dict(torch.load(f'./ckpts/resnet18_cifar100_epoch{latest_epoch}.pth'))
    #     print(f'Resuming training from epoch {latest_epoch}')

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # Learning rate scheduler with specific milestones for reduction
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    # Warmup scheduler
    iter_per_epoch = len(cifar100_train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch)

    os.makedirs('./ckpts', exist_ok=True)

    best_test_loss = float('inf')

    # Train the model
    for epoch in range(1, num_epochs + 1):
        if epoch > 1:
            scheduler.step(epoch)

        train_epoch(epoch, model, cifar100_train_loader, optimizer, criterion, device, warmup_scheduler)
        test_loss = test_epoch(epoch, model, cifar100_test_loader, criterion, device)

        # save every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'./ckpts/resnet18_cifar100_epoch{epoch}.pth')

        # Save the model with the best test loss
        if test_loss < best_test_loss:
            print(f'Find new best model at Epoch {epoch}')
            best_test_loss = test_loss
            torch.save(model.state_dict(), './ckpts/resnet18_cifar100_best.pth')

    return model, cifar100_test_loader


def transfer_linear_probe(model):
    """
    Transfer learning on CIFAR-10 using a linear probe
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

        test_loss = test_epoch(epoch, model, cifar10_test_loader, criterion, device)

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
    Extract features from the model
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
    Transfer learning on CIFAR-10 using a k-NN classifier
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


def replace_and_test_cifar100(model, test_loader, beta_vals):
    """
    Replace ReLU with BetaReLU and test the model on CIFAR-100
    """
    print('*' * 20)
    print('Running post-replace experiment on CIFAR-100...')
    print('*' * 20)
    criterion = nn.CrossEntropyLoss()

    # Test the original model
    print('Testing the original model...')
    test_epoch(-1, model, test_loader, criterion, device)

    # Test the model with different beta values
    for i, beta in enumerate(beta_vals):
        print(f'Using BetaReLU with beta={beta:.1f}')
        replacement_mapping = ReplacementMapping(beta=beta)
        orig_model = copy.deepcopy(model)
        new_model = replace_module(orig_model, replacement_mapping)
        test_epoch(-1, new_model, test_loader, criterion, device)


def main():
    # Initialize Weights and Biases (wandb)
    wandb.init(project='smooth-spline', entity='leyang_hu')

    # Train the model on CIFAR-100
    model, cifar100_test_loader = train()

    beta_vals = np.arange(0, 1, 0.1)

    # Replace ReLU with BetaReLU and test the model on CIFAR-100
    replace_and_test_cifar100(model, cifar100_test_loader, beta_vals)

    wandb.finish()


if __name__ == '__main__':
    main()


