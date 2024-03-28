# Your Name Here
# Short Description: This script adapts a pre-trained MNIST network to recognize Greek letters: alpha, beta, and gamma using transfer learning.

# Import statements
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image

# Class definitions

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

class GreekTransform:
    """Transforms for the Greek letters dataset to match MNIST format."""
    def __call__(self, x):
        x = TF.rgb_to_grayscale(x)
        x = TF.affine(x, angle=0, translate=(0, 0), scale=36/133, shear=0)
        x = TF.center_crop(x, output_size=(28, 28))
        x = TF.invert(x)
        return x

# Useful functions

def load_and_modify_network(model_path):
    """Loads the pre-trained model and modifies it for Greek letter classification."""
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path))
    for param in model.parameters():
        param.requires_grad = False
    # Replace the last layer with one suited for three classes (alpha, beta, gamma)
    num_features = model.fc1.out_features
    model.fc2 = nn.Linear(num_features, 3)
    return model

def train_network(model, train_loader, epochs=10):
    """Trains the network on the Greek letters dataset and plots the training error."""
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)
    model.train()
    losses = []  # List to store loss values
    
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss}')
    
    # Plotting the training loss
    plt.plot(losses, marker='o', linestyle='-', color='blue')
    plt.title('Training Loss Per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.show()

def prepare_dataloader(training_set_path):
    """Prepares the DataLoader for the Greek letters dataset with appropriate transformations."""
    transform = transforms.Compose([
        transforms.Resize((133, 133)),  # Resize images to enable the affine transformation
        transforms.ToTensor(),
        GreekTransform(),  # Apply custom transformations for Greek letters
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.ImageFolder(root=training_set_path, transform=transform)
    loader = DataLoader(dataset, batch_size=5, shuffle=True)
    return loader

def test_custom_images(model, test_images_directory):
    """Tests the model on all custom Greek letter images in the specified directory."""
    model.eval()  # Set the model to evaluation mode
    transform = transforms.Compose([
        transforms.Resize((133, 133)),  # Resize images for affine transformation
        transforms.ToTensor(),
        GreekTransform(),  # Custom transformations for Greek letters
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    class_labels = {0: 'alpha', 1: 'beta', 2: 'gamma'}
    images_paths = [os.path.join(test_images_directory, f) for f in os.listdir(test_images_directory) if os.path.isfile(os.path.join(test_images_directory, f))]

    plt.figure(figsize=(10, 10))
    for idx, image_path in enumerate(images_paths):
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = model(image_tensor)
            prediction = output.max(1, keepdim=True)[1]
            predicted_label = class_labels[prediction.item()]

        ax = plt.subplot(1, len(images_paths), idx + 1)
        plt.imshow(TF.to_pil_image(image_tensor.squeeze(0)), cmap='gray')
        plt.title(f'Predicted: {predicted_label}')
        plt.axis('off')

    plt.show()

def main(argv):
    """Main function handling the workflow."""
    if len(argv) < 4:
        print("Usage: python script_name.py <model_path> <training_set_path> <test_images_directory>")
        sys.exit(1)
    model_path = argv[1]
    training_set_path = argv[2]
    test_images_directory = argv[3]  # Path to the directory containing test images

    model = load_and_modify_network(model_path)
    train_loader = prepare_dataloader(training_set_path)
    train_network(model, train_loader)

    test_custom_images(model, test_images_directory)  # Test all images in the specified directory

if __name__ == "__main__":
    main(sys.argv)