import os
import cv2
import numpy as np 
import pandas as pd
from extract_keypoints import extract_keypoints

DATA_DIR = "data"
CATEGORIES = {
    "sitting": 0,
    "standing": 1
}

data = []

def build_dataset():
    # Extract data points for each image
    for category, label in CATEGORIES.items():
        folder = os.path.join(DATA_DIR, category)
        
        for filename in os.listdir(folder):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                path = os.path.join(folder, filename)
                image = cv2.imread(path)
                
                if image is  None:
                    print(f"Failed to load {filename}")
                    continue
            
            keypoints = extract_keypoints(image)
            if keypoints is not None:
                row = keypoints.tolist() + [label]
                data.append(row)
            else:
                print(f"No pose in {filename}")
                
        
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(data)
    df.to_csv("pose_dataset.csv", index=False)
    print(f"Saved dataset with {len(df)} samples to pose_dataset.csv")