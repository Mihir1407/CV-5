# Authors: Aditya Gurnani, Mihir Chitre
# This script is used to load a pre-trained model and predict handwritten digits from custom images using PyTorch.
# The network architecture is defined in the MyNetwork class and includes two convolutional layers, a dropout layer,
# and two fully connected layers. Live video feed is used to capture images for real-time digit prediction.

import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
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
    
# Loads the trained model from a file.
def load_model(model_path):
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()  
    return model

# Preprocesses the video frame for digit recognition.
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1 - x),  
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    frame = Image.fromarray(frame)
    return transform(frame)

# Predicts the class of a single image tensor using the provided model.
def predict_image(model, image_tensor):
    output = model(image_tensor.unsqueeze(0))  
    _, predicted = torch.max(output, 1)
    return predicted.item()

# Recognizes digits from a live video stream using the provided model.
def live_video_digit_recognition(model_path):
    model = load_model(model_path)
    cap = cv2.VideoCapture(0) 

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        processed_frame = preprocess_frame(frame)
        prediction = predict_image(model, processed_frame)

        cv2.putText(frame, f'Predicted Digit: {prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <model_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    live_video_digit_recognition(model_path)
