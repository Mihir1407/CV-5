# Your Name Here
# Short Description: Analyze and visualize the first layer filters of a trained PyTorch model.

# Import statements
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import cv2  # Import OpenCV
import numpy as np

# Assuming the MyNetwork class is defined in a separate module, import it here
# from my_network_module import MyNetwork

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

def load_model(model_path):
    """Loads the trained model from a file."""
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def analyze_first_layer(model):
    """Analyzes and visualizes the first layer filters."""
    weights = model.conv1.weight.data  # Access the weights of the first layer
    print("Shape of the first layer filters:", weights.shape)
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for i, ax in enumerate(axes.flat):
        if i >= weights.shape[0]:  # Only plot the filters, not all subplots may be used
            ax.axis('off')
            continue
        ax.imshow(weights[i, 0], interpolation='none')
        ax.set_title(f'Filter {i}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    
def apply_filters_and_visualize(model, train_loader):
    """Applies the first layer filters to the first training example and visualizes the results."""
    with torch.no_grad():  # Ensure gradients are not computed
        # Convert weights to NumPy array
        weights = model.conv1.weight.data.numpy()
        
        # Fetch the first image from the DataLoader
        dataiter = iter(train_loader)
        images, _ = next(dataiter)
        
        # Prepare the original image for filtering
        image = images[0].squeeze().numpy()

        # Setup plot
        fig, axes = plt.subplots(5, 4, figsize=(12, 15))
        
        for i in range(10):  # Assuming there are 10 filters
            col = i // 5 * 2  # Determine the column based on filter index (0 or 2)
            row = i % 5  # Row index
            
            # Plot original filter
            axes[row, col].imshow(weights[i, 0], cmap='gray', interpolation='none')
            # axes[row, col].set_title(f'Filter {i}')
            axes[row, col].axis('off')

            # Apply the filter using OpenCV's filter2D function
            filtered_image = cv2.filter2D(image, -1, weights[i, 0])
            
            # Plot filtered image
            axes[row, col + 1].imshow(filtered_image, cmap='gray', interpolation='none')
            # axes[row, col + 1].set_title(f'Filtered Image {i}')
            axes[row, col + 1].axis('off')

        plt.tight_layout()
        plt.show()

def main(argv):
    """Main function to execute the script logic."""
    if len(argv) < 2:
        print("Usage: python script_name.py <model_path>")
        sys.exit(1)  # Exit if the model path is not provided
    model_path = argv[1]
    model = load_model(model_path)
    
    analyze_first_layer(model)
    
    # Assuming the train_loader is defined and loaded as shown previously
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=1, shuffle=True)  # Load one image at a time
    
    apply_filters_and_visualize(model, train_loader)

if __name__ == "__main__":
    main(sys.argv)
