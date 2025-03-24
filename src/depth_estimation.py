"""
Date: 23.Mar.2025
Author: Harish Natarajan Ravi
Email: harrish.nr@gmail.com
"""

from transformers import pipeline
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
import os

# Load the model
checkpoint = "depth-anything/Depth-Anything-V2-base-hf"

# Device Configuration (Detect GPU if available, otherwise use CPU)
dev_conf = "cuda" if torch.cuda.is_available() else "cpu"
print("Device available:", dev_conf)

pipe = pipeline("depth-estimation", model=checkpoint, device=dev_conf)

# Ask user for input type
input_type = input("Enter 'url' to provide an image URL or 'local' to provide a local file path: ").strip().lower()

# Load the image based on input type
try:
    if input_type == 'url':
        url = input("Enter the image URL: ")
        image = Image.open(requests.get(url, stream=True).raw)
    elif input_type == 'local':
        path = input("Enter the local image file path: ")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        image = Image.open(path)
    else:
        raise ValueError("Invalid input type. Please enter 'url' or 'local'.")
except Exception as e:
    print(f"Failed to load image. Error: {e}")
    exit(1)

# Show original image
plt.figure(figsize=(7, 7))
plt.imshow(image)
plt.title("Input Image")
plt.axis('off')

# Predict depth
print("Running depth estimation...")
predictions = pipe(image)

# Get the depth image
pred_image = predictions["depth"]

# Show depth image
plt.figure(figsize=(7, 7))
plt.imshow(pred_image, cmap='inferno')
plt.title("Predicted Depth")
plt.axis('off')

plt.show()

print("Done!")