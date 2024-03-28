# Your Name Here
# Short Description: This script includes functionality to load the MNIST dataset using PyTorch and visualize the first six test digits.

# import statements
import sys
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        # Convolution layer with 10 5x5 filters
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # Convolution layer with 20 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # Dropout layer with a 0.5 dropout rate
        self.conv2_drop = nn.Dropout2d(p=0.5)
        # Fully connected layer with 50 nodes
        self.fc1 = nn.Linear(320, 50)
        # Final fully connected layer with 10 nodes
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # First conv layer + max pooling with 2x2 window + ReLU
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Second conv layer + dropout + max pooling with 2x2 window + ReLU
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # Flattening operation
        x = x.view(-1, 320)
        # First fully connected layer + ReLU
        x = F.relu(self.fc1(x))
        # Second (final) fully connected layer + log_softmax on the output
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Useful functions

def train_network(model, train_loader, test_loader, criterion, optimizer, epochs=5):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # Calculate training accuracy for this batch
            pred = output.data.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).sum().item()
            train_accuracies.append(correct / len(target))

        # Evaluate on test set
        test_loss, test_accuracy = test_model(model, test_loader, criterion)
        test_losses.extend([test_loss] * len(train_loader))  # Repeat to match train_losses length for plotting
        test_accuracies.extend([test_accuracy] * len(train_loader))  # Ditto for accuracies

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.4f} - Test Loss: {test_loss:.4f} - Train Accuracy: {train_accuracies[-1]:.4f} - Test Accuracy: {test_accuracy:.4f}')

    torch.save(model.state_dict(), 'mnist_model.pth')
    print('Model saved to mnist_model.pth')

    plot_errors(train_losses, test_losses, train_accuracies, test_accuracies, batch_size=64)

def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy

# Function to plot the training and testing error
def plot_errors(train_losses, test_losses, train_accuracies, test_accuracies, batch_size):
    num_batches = len(train_losses)
    examples_seen = [batch_size * i for i in range(num_batches)]

    # Plotting training and testing loss in a separate figure
    plt.figure(figsize=(12,6))  # Adjusted for wider and individual plot
    plt.plot(examples_seen, train_losses, 'b-', label='Train Loss', linewidth=1)
    plt.plot(examples_seen, test_losses, 'r-', label='Test Loss', linewidth=1)
    plt.xlabel('Number of Training Examples Seen')
    plt.ylabel('Negative Log Likelihood Loss')
    plt.title('Training and Testing Loss')
    plt.xticks([i * 25000 for i in range((num_batches * batch_size) // 25000 + 1)])
    plt.legend()
    plt.show()  # Show the first plot before moving on to the next

    # Plotting training and testing accuracy in a separate figure
    plt.figure(figsize=(12,6))  # Adjusted for wider and individual plot
    plt.plot(examples_seen, train_accuracies, 'g-', label='Train Accuracy', linewidth=1)
    plt.plot(examples_seen, test_accuracies, 'orange', label='Test Accuracy', linewidth=1)
    plt.xlabel('Number of Training Examples Seen')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.xticks([i * 25000 for i in range((num_batches * batch_size) // 25000 + 1)])
    plt.legend()
    plt.show()  # Show the second plot
    
# Function to visualize the first six digits of the MNIST test set
def visualize_test_digits():
    """
    Loads the MNIST test dataset and visualizes the first six digits.
    """
    # Load the MNIST test dataset
    test_transform = transforms.Compose([transforms.ToTensor()])
    testset = MNIST(root='./data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=6, shuffle=False)

    # Get a batch of test images
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # Create a grid of plots
    plt.figure(figsize=(17, 8))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i].numpy().squeeze(), cmap='gray')
        plt.title(f"Label: {labels[i].item()}")
    plt.show()

# Main function
def main(argv):
    """
    Main function to handle workflow of loading and visualizing MNIST dataset.
    """
    # Handling command line arguments (Placeholder for argument handling)

    # Visualize the first six digits of the MNIST test set
    visualize_test_digits()

    # Placeholder for additional code (e.g., network initialization, training)
    # Initialize the network
    model = MyNetwork()

    # Define the loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Load datasets
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    testset = MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
    test_loader = DataLoader(testset, batch_size=1000, shuffle=False)

    # Call the training function
    train_network(model, train_loader, test_loader, criterion, optimizer)

if __name__ == "__main__":
    main(sys.argv)
