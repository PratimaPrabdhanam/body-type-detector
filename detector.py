from PIL import Image
import io
import numpy as np
import mediapipe as mp
import cv2
from utils.model_utils import load_trained_model, predict_body_shape
from utils.data_loader import process_live_keypoints

# Load model and body shapes once
model, body_shapes = load_trained_model("body_detection_model.h5")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def predict_body_type(image: Image.Image) -> str:
    """
    Core logic to predict body type from a PIL Image.
    """
    image_np = np.array(image)
    results = pose.process(image_np)

    if results.pose_landmarks:
        keypoints = process_live_keypoints(results.pose_landmarks)
        body_shape = predict_body_shape(model, keypoints, body_shapes)
        return body_shape
    else:
        return "Unknown"

def detect_from_bytes(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return predict_body_type(image)

def detect_from_cv2_frame(frame: np.ndarray) -> str:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return predict_body_type(image)
