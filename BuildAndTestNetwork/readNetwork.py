# Authors: Aditya Gurnani, Mihir Chitre
# Short Description: This script loads a trained PyTorch model and tests it on the first 10 examples of the MNIST test set.
# It demonstrates loading a model, performing inference on a subset of the MNIST dataset, and visualizing the results.
# The network architecture, defined in the MyNetwork class, includes convolutional layers, a dropout layer, and fully connected layers.

import sys
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# Defines the architecture of a convolutional neural network for digit classification.
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

# Load the pre-trained model from the specified path
def load_model(model_path):
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()  
    return model

# Test the model on the first 10 examples from the MNIST test dataset
def test_on_examples(model, test_loader):
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    outputs = model(images[:10])
    _, predicted = torch.max(outputs, 1)

    for idx in range(10):
        print(f"Outputs: {outputs[idx].data.numpy().round(2)}, Predicted: {predicted[idx].item()}, Actual: {labels[idx].item()}")

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for i, ax in enumerate(axes.flat):
        if i >= 9:
            break
        ax.imshow(images[i].numpy().squeeze(), cmap='gray', interpolation='none')
        ax.set_title(f"Predicted: {predicted[i].item()}")
        ax.axis('off')
    plt.show()

# Main function to orchestrate the model loading and testing
def main(argv):
    if len(argv) < 2:
        print("Usage: python script_name.py <model_path>")
        sys.exit(1)
    model_path = argv[1]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    testset = MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=10, shuffle=False)

    model = load_model(model_path)
    test_on_examples(model, test_loader)

if __name__ == "__main__":
    main(sys.argv)