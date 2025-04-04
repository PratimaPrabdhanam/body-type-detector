import cv2
import numpy as np
import mediapipe as mp
from utils.model_utils import load_trained_model, predict_body_shape
from utils.data_loader import process_live_keypoints

# Load model
model, body_shapes = load_trained_model("body_detection_model.h5")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Start webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        keypoints = process_live_keypoints(results.pose_landmarks)
        body_shape = predict_body_shape(model, keypoints, body_shapes)
        print(f"Predicted Body Shape: {body_shape}")  # Debugging line

        # Display body shape on frame
        cv2.putText(frame, f"Body Shape: {body_shape}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Live Body Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
