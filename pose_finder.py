from joblib import load
import numpy as np
from extract_keypoints import extract_keypoints
import os
import cv2

# Load the model
model = load('model/rf_pose_classifier.joblib')

# Image path
image_path = "test.jpeg"

# Check if file exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file '{image_path}' not found.")

# Try loading the image
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Failed to load image '{image_path}'. The file may be corrupted or not a valid image.")

# Extract keypoints
keypoints = extract_keypoints(image)

# Predict
if keypoints is None:
    raise ValueError("No keypoints detected")

prediction = model.predict(keypoints.reshape(1, -1))

# Interpret result
if prediction[0] == 0:
    print("Sitting")
else:
    print("Standing")
