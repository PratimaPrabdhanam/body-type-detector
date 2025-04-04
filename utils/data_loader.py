import json
import numpy as np


def load_dataset(data_path):
    """ Load dataset from a JSON file containing keypoints and labels. """
    with open(data_path, 'r') as f:
        data = json.load(f)

    X = np.array([np.array(sample['keypoints']) for sample in data])  # Extract keypoints
    y = np.array([sample['label'] for sample in data])  # Extract labels

    return X, y


def process_live_keypoints(pose_landmarks):
    """ Convert live pose landmarks into a structured format (1D array). """
    keypoints = []

    # Extract x, y, and z coordinates from all 33 keypoints
    for landmark in pose_landmarks.landmark:
        keypoints.extend([landmark.x, landmark.y, landmark.z])

    return np.array(keypoints).reshape(1, -1)  # Reshape to (1, 99)
