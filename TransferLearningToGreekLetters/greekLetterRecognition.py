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
from torchviz import make_dot

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

# def prepare_dataloader(training_set_path, batch_size=64):
#     transform = transforms.Compose([
#         transforms.Resize((133, 133)),
#         transforms.RandomRotation(degrees=15),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         GreekTransform(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])
#     dataset = datasets.ImageFolder(root=training_set_path, transform=transform)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     return loader

# Useful functions

def load_and_modify_network(model_path):
    """Loads the pre-trained model and modifies it for Greek letter classification."""
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path))
    for param in model.parameters():
        param.requires_grad = False
    # Replace the last layer with one suited for three classes (alpha, beta, gamma)
    num_features = model.fc1.out_features
    model.fc2 = nn.Linear(num_features, 5)
    return model

def train_network(model, train_loader, epochs=30):
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # criterion = nn.NLLLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)
    criterion = nn.NLLLoss()  # Assuming you're using NLLLoss as your criterion
    model.train()
    
    losses = []  # Store loss values
    accuracies = []  # Store accuracies

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

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        # scheduler.step()
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        accuracy = 100. * correct / len(train_loader.dataset)
        accuracies.append(accuracy)  # Append accuracy after each epoch

        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Call to the new plotting function
    plot_training_metrics(losses, accuracies, epochs)

def plot_training_metrics(losses, accuracies, epochs):
    """
    Plots the training loss and accuracy over epochs.

    Parameters:
    - losses: List of average loss values per epoch.
    - accuracies: List of accuracy percentages per epoch.
    - epochs: Total number of epochs.
    """
    plt.figure(figsize=(12, 6))

    # Plotting training loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), losses, marker='o', linestyle='-', color='blue')
    plt.title('Training Loss Per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')

    # Plotting training accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), accuracies, marker='o', linestyle='-', color='red')
    plt.title('Training Accuracy Per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    plt.tight_layout()
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

    class_labels = {0: 'alpha', 1: 'beta', 2: 'gamma', 3: 'lambda', 4: 'theta'}
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
    
def visualize_model_architecture(model):
    """
    Visualizes the architecture of a given PyTorch model.
    
    Parameters:
    - model: The PyTorch model to visualize.
    """
    # Dummy input that matches the input dimensions expected by the model
    # For MyNetwork, it expects a single-channel (grayscale) image of 28x28 pixels
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # Forward pass through the model to get the output
    model_output = model(dummy_input)
    
    # Generate the graph
    dot = make_dot(model_output, params=dict(list(model.named_parameters()) + [('input', dummy_input)]))
    
    # Render the graph to visualize it
    dot.render('model_architecture', format='png', cleanup=True)
    print("Model architecture saved as 'model_architecture.png'.")

def main(argv):
    """Main function handling the workflow."""
    if len(argv) < 4:
        print("Usage: python script_name.py <model_path> <training_set_path> <test_images_directory>")
        sys.exit(1)
    model_path = argv[1]
    training_set_path = argv[2]
    test_images_directory = argv[3]  # Path to the directory containing test images

    model = load_and_modify_network(model_path)
    visualize_model_architecture(model)
    
    train_loader = prepare_dataloader(training_set_path)
    train_network(model, train_loader)

    test_custom_images(model, test_images_directory)  # Test all images in the specified directory

if __name__ == "__main__":
    main(sys.argv)