import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# Assuming MyNetwork and other utility functions like load_model are defined as before
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

def preprocess_frame(frame):
    """Preprocesses the video frame for digit recognition."""
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1 - x),  # Inverting colors
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # Convert frame to PIL Image to use with torchvision transforms
    frame = Image.fromarray(frame)
    return transform(frame)

def predict_image(model, image_tensor):
    output = model(image_tensor.unsqueeze(0))  # Add batch dimension
    _, predicted = torch.max(output, 1)
    return predicted.item()

def live_video_digit_recognition(model_path):
    """Recognizes digits from a live video stream."""
    model = load_model(model_path)
    cap = cv2.VideoCapture(0)  # 0 for the default camera

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Preprocess the captured frame
        processed_frame = preprocess_frame(frame)

        # Predict the digit
        prediction = predict_image(model, processed_frame)

        # Display the resulting frame with the predicted digit
        cv2.putText(frame, f'Predicted Digit: {prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <model_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    live_video_digit_recognition(model_path)
