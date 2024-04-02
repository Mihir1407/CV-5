# Authors: Aditya Gurnani, Mihir Chitre
# Short Description: This script adapts a pre-trained MNIST network to recognize Greek letters: alpha, beta, and gamma using transfer learning.

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
from torchviz import make_dot

# Defines the architecture of a CNN initially trained for digit classification.
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
    
# Custom transformations for Greek letter dataset images to match the MNIST format.
class GreekTransform:
    def __call__(self, x):
        x = TF.rgb_to_grayscale(x)
        x = TF.affine(x, angle=0, translate=(0, 0), scale=36/133, shear=0)
        x = TF.center_crop(x, output_size=(28, 28))
        x = TF.invert(x)
        return x

# Loads the pre-trained model and modifies it for Greek letter classification.
def load_and_modify_network(model_path):
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path))
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc1.out_features
    model.fc2 = nn.Linear(num_features, 5)
    return model

# Trains the adapted network on a new dataset, using SGD optimizer and NLLLoss.
def train_network(model, train_loader, epochs=30):
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)
    criterion = nn.NLLLoss()  
    model.train()
    
    losses = []  
    accuracies = []  

    for epoch in range(epochs):
        total_loss = 0
        correct = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        accuracy = 100. * correct / len(train_loader.dataset)
        accuracies.append(accuracy)  
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    plot_training_metrics(losses, accuracies, epochs)

"""
    Plots the training loss and accuracy over epochs.

    Parameters:
    - losses: List of average loss values per epoch.
    - accuracies: List of accuracy percentages per epoch.
    - epochs: Total number of epochs.
    """
def plot_training_metrics(losses, accuracies, epochs):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), losses, marker='o', linestyle='-', color='blue')
    plt.title('Training Loss Per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), accuracies, marker='o', linestyle='-', color='red')
    plt.title('Training Accuracy Per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    plt.tight_layout()
    plt.show()

# Prepares the DataLoader for the Greek letters dataset with appropriate transformations.
def prepare_dataloader(training_set_path):
    transform = transforms.Compose([
        transforms.Resize((133, 133)),  
        transforms.ToTensor(),
        GreekTransform(),  
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.ImageFolder(root=training_set_path, transform=transform)
    loader = DataLoader(dataset, batch_size=5, shuffle=True)
    return loader

# Tests the model on all custom Greek letter images in the specified directory.
def test_custom_images(model, test_images_directory):
    model.eval()  
    transform = transforms.Compose([
        transforms.Resize((133, 133)),  
        transforms.ToTensor(),
        GreekTransform(),  
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    class_labels = {0: 'alpha', 1: 'beta', 2: 'gamma', 3: 'lambda', 4: 'theta'}
    images_paths = [os.path.join(test_images_directory, f) for f in os.listdir(test_images_directory) if os.path.isfile(os.path.join(test_images_directory, f))]

    plt.figure(figsize=(10, 10))
    for idx, image_path in enumerate(images_paths):
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0)  
        with torch.no_grad():
            output = model(image_tensor)
            prediction = output.max(1, keepdim=True)[1]
            predicted_label = class_labels[prediction.item()]

        ax = plt.subplot(1, len(images_paths), idx + 1)
        plt.imshow(TF.to_pil_image(image_tensor.squeeze(0)), cmap='gray')
        plt.title(f'Predicted: {predicted_label}')
        plt.axis('off')

    plt.show()
    
"""
    Visualizes the architecture of a given PyTorch model.
    
    Parameters:
    - model: The PyTorch model to visualize.
"""
def visualize_model_architecture(model):
    dummy_input = torch.randn(1, 1, 28, 28)
    
    model_output = model(dummy_input)
    
    dot = make_dot(model_output, params=dict(list(model.named_parameters()) + [('input', dummy_input)]))
    
    dot.render('model_architecture', format='png', cleanup=True)
    print("Model architecture saved as 'model_architecture.png'.")

# Main execution function: loads model, modifies it, re-trains, and tests on custom images.
def main(argv):
    if len(argv) < 4:
        print("Usage: python script_name.py <model_path> <training_set_path> <test_images_directory>")
        sys.exit(1)
    model_path = argv[1]
    training_set_path = argv[2]
    test_images_directory = argv[3]  

    model = load_and_modify_network(model_path)
    visualize_model_architecture(model)
    
    train_loader = prepare_dataloader(training_set_path)
    train_network(model, train_loader)

    test_custom_images(model, test_images_directory)  

if __name__ == "__main__":
    main(sys.argv)