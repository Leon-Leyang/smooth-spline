import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
from torchvision.models import resnet18
from utils import WarmUpLR


def get_data_loaders(batch_size=128):
    """
    Get the CIFAR100 data loaders
    """
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
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
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
    wandb.log({'train_loss': train_loss, 'train_accuracy': train_accuracy})


def test_epoch(epoch, model, testloader, criterion, device):
    """
    Test the model for one epoch
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
    print(f'Test Epoch {epoch}, Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')

    # Log the test loss and accuracy to wandb
    wandb.log({'val_loss': test_loss, 'val_accuracy': test_accuracy})

    return test_loss


def train():
    """
    Train and return the model
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Hyperparameters
    batch_size = 128
    learning_rate = 0.1
    num_epochs = 200

    # Get the data loaders
    train_loader, test_loader = get_data_loaders(batch_size)

    # Initialize the model
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # Learning rate scheduler with specific milestones for reduction
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    # Warmup scheduler
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch)

    os.makedirs('./ckpts', exist_ok=True)

    best_test_loss = float('inf')

    # Train the model
    for epoch in range(1, num_epochs + 1):
        if epoch > 1:
            scheduler.step()

        train_epoch(epoch, model, train_loader, optimizer, criterion, device, warmup_scheduler)
        test_loss = test_epoch(epoch, model, test_loader, criterion, device)

        # save every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'./ckpts/resnet18_cifar100_epoch{epoch}.pth')

        # Save the model with the best test loss
        if test_loss < best_test_loss:
            print(f'Find new best model at Epoch {epoch}')
            best_test_loss = test_loss
            torch.save(model.state_dict(), './ckpts/resnet18_cifar100_best.pth')

    wandb.finish()

    return model


def main():
    # Initialize Weights and Biases (wandb)
    wandb.init(project='smooth-spline', entity='leyang_hu')

    model = train()


if __name__ == '__main__':
    main()


