# Authors: Aditya Gurnani, Mihir Chitre
# This script includes functionality to load a pre-trained convolutional neural network
# using PyTorch, predict handwritten digits from custom images, and visualize the predictions.

import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

"""
    Defines the architecture of a convolutional neural network for digit classification.
    The network consists of two convolutional layers, a dropout layer, and two fully connected layers.
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

# Loads the trained model from a file.
def load_model(model_path):
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()  
    return model

# Converts images to grayscale, resizes, inverts colors, and normalizes.
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),  
        transforms.Lambda(lambda x: 1 - x),  
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    image = Image.open(image_path)
    return transform(image)

# Runs a forward pass on an image to predict the digit.
def predict_image(model, image_tensor):
    output = model(image_tensor.unsqueeze(0))  
    _, predicted = torch.max(output, 1)
    return predicted.item()

# Displays images alongside their predicted labels.
def visualize_predictions(image_tensors, predictions):
    plt.figure(figsize=(10, 4))
    for i, (image_tensor, prediction) in enumerate(zip(image_tensors, predictions)):
        image = image_tensor.squeeze().numpy()  
        plt.subplot(2, 5, i+1)
        plt.imshow(image, cmap='gray', vmin=0, vmax=1)  
        plt.title(f'Predicted: {prediction}')
        plt.axis('off')
    plt.show()

# Processes all images in a directory and predicts their labels using a trained model.
def process_images_from_directory(directory_path, model_path): 
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

# Main function that orchestrates the model evaluation on custom digit images.
def main(argv):
    if len(argv) < 3:
        print("Usage: python script_name.py <directory_path> <model_path>")
        return
    directory_path = argv[1]
    model_path = argv[2]
    process_images_from_directory(directory_path, model_path)

if __name__ == "__main__":
    main(sys.argv)
