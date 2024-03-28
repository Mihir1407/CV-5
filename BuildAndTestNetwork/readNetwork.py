# Your Name Here
# Short Description: This script loads a trained PyTorch model and tests it on the first 10 examples of the MNIST test set.

# Import statements
import sys
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

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

def test_on_examples(model, test_loader):
    """Tests the model on the first 10 examples from the MNIST test set."""
    dataiter = iter(test_loader)
    images, labels = next(dataiter)  # Use next() function here
    outputs = model(images[:10])
    _, predicted = torch.max(outputs, 1)

    # Display model outputs, predictions, and actual labels
    for idx in range(10):
        print(f"Outputs: {outputs[idx].data.numpy().round(2)}, Predicted: {predicted[idx].item()}, Actual: {labels[idx].item()}")

    # Plot the first 9 digits
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for i, ax in enumerate(axes.flat):
        if i >= 9:  # Only plot the first 9 images
            break
        ax.imshow(images[i].numpy().squeeze(), cmap='gray', interpolation='none')
        ax.set_title(f"Predicted: {predicted[i].item()}")
        ax.axis('off')
    plt.show()

def main(argv):
    """Main function to execute the script logic."""
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
