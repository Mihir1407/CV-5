# Authors: Aditya Gurnani, Mihir Chitre
# This script is used to load the MNIST dataset using PyTorch, train a simple convolutional neural network on the data,
# and visualize the first six test digits. The network architecture is defined in the MyNetwork class and includes two
# convolutional layers, a dropout layer, and two fully connected layers. 

import sys
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchviz import make_dot

#This class defines the architecture of a convolutional neural network for digit classification.
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Function to train the network, plot training and test errors, and accuracies.
def train_network(model, train_loader, test_loader, criterion, optimizer, epochs=5):
    train_losses = []
    train_accuracies = []  
    train_counter = []
    test_losses = []
    test_accuracies = []  
    test_counter = [i*len(train_loader.dataset) for i in range(epochs + 1)]

    test_loss, test_accuracy = test_model(model, test_loader, criterion)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    log_interval = 10
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1, keepdim=True) 
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = correct / len(data)

            if batch_idx % log_interval == 0:
                train_losses.append(loss.item())
                train_accuracies.append(accuracy)  # Append accuracy
                examples_seen = epoch * len(train_loader.dataset) + batch_idx * len(data)
                train_counter.append(examples_seen)

        test_loss, test_accuracy = test_model(model, test_loader, criterion)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    plot_combined_errors(train_counter, train_losses, test_counter, test_losses, "Loss")
    plot_combined_errors(train_counter, train_accuracies, test_counter, test_accuracies, "Accuracy")

# Function to evaluate the model on test data and calculate the loss and accuracy.
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
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

# Function to plot training and testing metrics (either losses or accuracies) on the same graph.
def plot_combined_errors(train_counter, train_metric, test_counter, test_metric, metric_name):
    fig = plt.figure()
    plt.plot(train_counter, train_metric, color='blue')
    plt.scatter(test_counter, test_metric, color='red')
    plt.legend([f'Train {metric_name}', f'Test {metric_name}'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel(metric_name)
    plt.title(f'Train vs Test {metric_name}')
    plt.show()
    
# Function to visualize the first six digits of the MNIST test set
def visualize_test_digits():
    test_transform = transforms.Compose([transforms.ToTensor()])
    testset = MNIST(root='./data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=6, shuffle=False)

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    plt.figure(figsize=(17, 8))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i].numpy().squeeze(), cmap='gray')
        plt.title(f"Label: {labels[i].item()}")
    plt.show()
    
# Visualizes the architecture of a given PyTorch model.    
def visualize_model_architecture(model):
    dummy_input = torch.randn(1, 1, 28, 28)
    
    model_output = model(dummy_input)
    
    dot = make_dot(model_output, params=dict(list(model.named_parameters()) + [('input', dummy_input)]))
    
    dot.render('model_architecture', format='png', cleanup=True)
    print("Model architecture saved as 'model_architecture.png'.")


# Main function to handle workflow of loading and visualizing MNIST dataset.
def main(argv):
    visualize_test_digits()
    model = MyNetwork()
    criterion = nn.NLLLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    testset = MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
    test_loader = DataLoader(testset, batch_size=1000, shuffle=False)

    visualize_model_architecture(model)
    train_network(model, train_loader, test_loader, criterion, optimizer)

if __name__ == "__main__":
    main(sys.argv)
