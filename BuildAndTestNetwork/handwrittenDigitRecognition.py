# Your Name Here
# A script to load a pre-trained model and predict handwritten digits from custom images.

import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# Class Definitions
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

# Useful Functions

def load_model(model_path):
    """Loads the trained model from a file."""
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_image(image_path):
    """Converts images to grayscale, resizes, inverts colors, and normalizes."""
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),  # Converts image to tensor and scales to [0, 1]
        transforms.Lambda(lambda x: 1 - x),  # Invert colors: now it operates on the tensor
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    image = Image.open(image_path)
    return transform(image)

def predict_image(model, image_tensor):
    """Runs a forward pass on an image to predict the digit."""
    output = model(image_tensor.unsqueeze(0))  # Add batch dimension
    _, predicted = torch.max(output, 1)
    return predicted.item()

def visualize_predictions(image_tensors, predictions):
    """Displays images alongside their predicted labels."""
    plt.figure(figsize=(10, 4))
    for i, (image_tensor, prediction) in enumerate(zip(image_tensors, predictions)):
        image = image_tensor.squeeze().numpy()  # Remove batch dim and convert to numpy
        plt.subplot(2, 5, i+1)
        plt.imshow(image, cmap='gray', vmin=0, vmax=1)  # Adjusted for normalized tensor
        plt.title(f'Predicted: {prediction}')
        plt.axis('off')
    plt.show()

def process_images_from_directory(directory_path, model_path):
    """Processes all images in a directory and predicts their labels using a trained model."""
    model = load_model(model_path)
    image_paths = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path)]
    image_tensors = []
    predictions = []

    for image_path in image_paths:
        image_tensor = preprocess_image(image_path)
        prediction = predict_image(model, image_tensor)
        predictions.append(prediction)
        image_tensors.append(image_tensor)

    visualize_predictions(image_tensors, predictions)

# Main Function
def main(argv):
    """Main function that orchestrates the model evaluation on custom digit images."""
    # Assuming argv[1] is the directory path and argv[2] is the model path
    if len(argv) < 3:
        print("Usage: python script_name.py <directory_path> <model_path>")
        return
    directory_path = argv[1]
    model_path = argv[2]
    process_images_from_directory(directory_path, model_path)

if __name__ == "__main__":
    main(sys.argv)
