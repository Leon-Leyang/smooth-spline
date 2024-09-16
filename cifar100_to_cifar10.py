import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
from torchvision.models import resnet18


# Initialize Weights and Biases (wandb)
wandb.init(project='smooth-spline', entity='leyang_hu')

# Hyperparameters
batch_size = 128
learning_rate = 0.1
num_epochs = 200

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data Preparation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Model: ResNet18
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 100)  # CIFAR-100 has 100 classes
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True)

# Learning rate scheduler with specific milestones for reduction
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)


# Training function
def train(epoch):
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

    # Log the training loss and accuracy to wandb
    wandb.log({'train_loss': train_loss, 'train_accuracy': train_accuracy})


# Testing function
def test(epoch):
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


# Main loop
for epoch in range(num_epochs):
    train(epoch)
    test(epoch)
    scheduler.step()

# Save the final model
os.makedirs('./ckpts', exist_ok=True)
torch.save(model.state_dict(), './ckpts/resnet18_cifar100.pth')

# Mark the wandb run as complete
wandb.finish()
