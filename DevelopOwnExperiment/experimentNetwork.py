# import necessary libraries
import sys
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST as MNIST  # FashionMNIST for a challenge
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
import time

class ExperimentNetwork(nn.Module):
    def __init__(self, conv_layers=2, num_filters=10, dropout_rate=0.5, fc_nodes=50):
        super(ExperimentNetwork, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.feature_size = 28  # Initial size of the image
        for i in range(conv_layers):
            in_channels = 1 if i == 0 else num_filters
            self.conv_layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=5, padding=2))
            # After each conv layer, apply ReLU (no change in size) and then MaxPool which halves the dimensions
            self.feature_size //= 2  # MaxPool2d with kernel_size=2
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate the size dynamically based on the conv_layers and pooling
        fc_input_size = num_filters * self.feature_size * self.feature_size
        self.fc1 = nn.Linear(fc_input_size, fc_nodes)  # Now using the fc_nodes parameter
        self.fc2 = nn.Linear(fc_nodes, 10)

    def forward(self, x):
        for layer in self.conv_layers:
            x = F.relu(F.max_pool2d(layer(x), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train_and_evaluate(model, train_loader, test_loader, epochs=1):
     # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.NLLLoss()
    model.train()
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

def run_experiments():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    testset = MNIST(root='./data', train=False, download=True, transform=transform)

    # Adjusted ranges for achieving at least 100 variations with around 25 values for epochs and dropout rates
    conv_layers_options = [1, 2, 3, 4]
    epochs_options = list(range(1, 26, 1))
    dropout_rates_options = [x * 0.04 for x in range(1, 26)]
    fc_nodes_options = [50, 100, 150]
    batch_size_options = [32, 64, 128]

    results = []
    variation_count = 0

    fixed_epochs = 5
    fixed_dropout_rate = 0.3
    fixed_batch_size = 64 
    fixed_fc_nodes = 50
    fixed_conv_layers = 2
    train_loader = DataLoader(trainset, batch_size=fixed_batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=fixed_batch_size, shuffle=False, num_workers=4)
    
    for conv_layers in conv_layers_options:
        if variation_count >= 100: break
        model = ExperimentNetwork(conv_layers=conv_layers, num_filters=10, dropout_rate=0.3, fc_nodes=fixed_fc_nodes)
        accuracy = train_and_evaluate(model, train_loader, test_loader, epochs=5)
        results.append((conv_layers, 5, 0.3, fixed_fc_nodes, fixed_batch_size, accuracy))
        variation_count += 1
        print(f"Variation {variation_count}: Conv Layers: {conv_layers}, Epochs: {fixed_epochs}, Dropout: {fixed_dropout_rate}, "
              f"FC Nodes: {fixed_fc_nodes}, Batch Size: {fixed_batch_size}, Accuracy: {accuracy:.2f}%")

    for epochs in epochs_options:
        if variation_count >= 100: break
        model = ExperimentNetwork(conv_layers=2, num_filters=10, dropout_rate=0.3, fc_nodes=fixed_fc_nodes)
        accuracy = train_and_evaluate(model, train_loader, test_loader, epochs=epochs)
        results.append((2, epochs, 0.3, fixed_fc_nodes, fixed_batch_size, accuracy))
        variation_count += 1
        print(f"Variation {variation_count}: Conv Layers: {fixed_conv_layers}, Epochs: {epochs}, Dropout: {fixed_dropout_rate}, "
              f"FC Nodes: {fixed_fc_nodes}, Batch Size: {fixed_batch_size}, Accuracy: {accuracy:.2f}%")

    for dropout_rate in dropout_rates_options:
        if variation_count >= 100: break
        model = ExperimentNetwork(conv_layers=2, num_filters=10, dropout_rate=dropout_rate, fc_nodes=fixed_fc_nodes)
        accuracy = train_and_evaluate(model, train_loader, test_loader, epochs=5)
        results.append((2, 5, dropout_rate, fixed_fc_nodes, fixed_batch_size, accuracy))
        variation_count += 1
        print(f"Variation {variation_count}: Conv Layers: {fixed_conv_layers}, Epochs: {fixed_epochs}, Dropout: {dropout_rate}, "
              f"FC Nodes: {fixed_fc_nodes}, Batch Size: {fixed_batch_size}, Accuracy: {accuracy:.2f}%")
            
    if variation_count < 100: 
        for fc_nodes in fc_nodes_options:
            if variation_count >= 100: break
            model = ExperimentNetwork(conv_layers=fixed_conv_layers, num_filters=10, 
                                      dropout_rate=fixed_dropout_rate, fc_nodes=fc_nodes)
            accuracy = train_and_evaluate(model, train_loader, test_loader, epochs=fixed_epochs)
            results.append((fixed_conv_layers, fixed_epochs, fixed_dropout_rate, fc_nodes, fixed_batch_size, accuracy))
            variation_count += 1
            print(f"Variation {variation_count}: Conv Layers: {fixed_conv_layers}, FC Nodes: {fc_nodes}, "
                  f"Epochs: {fixed_epochs}, Dropout: {fixed_dropout_rate}, Batch Size: {fixed_batch_size}, "
                  f"Accuracy: {accuracy:.2f}%")
    
    if variation_count < 100: 
        for batch_size in batch_size_options:
            if variation_count >= 100: break
            train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
            test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
            model = ExperimentNetwork(conv_layers=fixed_conv_layers, num_filters=10, 
                                      dropout_rate=fixed_dropout_rate, fc_nodes=fixed_fc_nodes)
            accuracy = train_and_evaluate(model, train_loader, test_loader, epochs=fixed_epochs)
            results.append((fixed_conv_layers, fixed_epochs, fixed_dropout_rate, fixed_fc_nodes, batch_size, accuracy))
            variation_count += 1
            print(f"Variation {variation_count}: Conv Layers: {fixed_conv_layers}, FC Nodes: {fixed_fc_nodes}, "
                  f"Epochs: {fixed_epochs}, Dropout: {fixed_dropout_rate}, Batch Size: {batch_size}, "
                  f"Accuracy: {accuracy:.2f}%")

    results.sort(key=lambda x: x[-1], reverse=True)
    
    best_config = results[0]
    print(f"Best Configuration: Conv Layers: {best_config[0]}, Epochs: {best_config[1]}, "
          f"Dropout: {best_config[2]}, FC Nodes: {best_config[3]}, "
          f"Batch Size: {best_config[4]}, Accuracy: {best_config[5]:.2f}%")
    
    # Print the worst configuration (last in the sorted list)
    worst_config = results[-1]
    print(f"Worst Configuration: Conv Layers: {worst_config[0]}, Epochs: {worst_config[1]}, "
          f"Dropout: {worst_config[2]}, FC Nodes: {worst_config[3]}, "
          f"Batch Size: {worst_config[4]}, Accuracy: {worst_config[5]:.2f}%")

if __name__ == "__main__":
    run_experiments()
