# Authors: Aditya Gurnani, Mihir Chitre
# Short Description: This script loads a trained PyTorch model and analyzes & visualizes the first convolutional layer's filters.
# It displays the filters' weights as images and applies each filter to the first image of the MNIST dataset to observe their effects.

import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import cv2  # Import OpenCV
import numpy as np

"""
    Defines a convolutional neural network architecture with configurable layers and parameters.
    Designed for digit classification on the MNIST dataset.
"""
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

# Loads a trained model from a specified path for further analysis or inference.
def load_model(model_path):
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval() 
    return model

# Visualizes the weights of the first convolutional layer as images to understand learned patterns.
def analyze_first_layer(model):
    weights = model.conv1.weight.data  
    print("Shape of the first layer filters:", weights.shape)
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for i, ax in enumerate(axes.flat):
        if i >= weights.shape[0]:  
            ax.axis('off')
            continue
        ax.imshow(weights[i, 0], interpolation='none')
        ax.set_title(f'Filter {i}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    
"""
    Applies each filter of the first convolutional layer to the first image in the MNIST dataset and visualizes the output.
    Demonstrates the effect of each filter on detecting features from the input image.
"""
def apply_filters_and_visualize(model, train_loader):
    with torch.no_grad():  
        weights = model.conv1.weight.data.numpy()
        
        dataiter = iter(train_loader)
        images, _ = next(dataiter)
        
        image = images[0].squeeze().numpy()

        fig, axes = plt.subplots(5, 4, figsize=(12, 15))
        
        for i in range(10):  
            col = i // 5 * 2  
            row = i % 5  
            
            axes[row, col].imshow(weights[i, 0], cmap='gray', interpolation='none')
            axes[row, col].axis('off')

            filtered_image = cv2.filter2D(image, -1, weights[i, 0])
            
            axes[row, col + 1].imshow(filtered_image, cmap='gray', interpolation='none')
            axes[row, col + 1].axis('off')

        plt.tight_layout()
        plt.show()

# Executes the script: loads the model, visualizes its first layer filters, and applies those filters to an MNIST image.
def main(argv):
    if len(argv) < 2:
        print("Usage: python script_name.py <model_path>")
        sys.exit(1)  
    model_path = argv[1]
    model = load_model(model_path)
    
    analyze_first_layer(model)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=1, shuffle=True)  
    
    apply_filters_and_visualize(model, train_loader)

if __name__ == "__main__":
    main(sys.argv)
