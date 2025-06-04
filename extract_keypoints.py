import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def extract_keypoints(image):
    """
    Given an image (BGR), return a flattened array of pose keypoints (x, y).
    If no pose is detected, returns None.
    """
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        return None
    
    keypoints = []
    for lm in results.pose_landmarks.landmark:
        keypoints.extend([lm.x, lm.y])
        
    return np.array(keypoints)