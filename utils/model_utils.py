import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


# Load the trained model
def load_trained_model(model_path):
    """Loads the trained body shape detection model."""
    model = load_model(model_path)

    # Define the body shape categories (Adjust these as per your training labels)
    body_shapes = ["pear", "rectangle", "hourglass", "apple", "inverted_triangle"]

    return model, body_shapes


# Predict body shape based on keypoints
def predict_body_shape(model, keypoints, body_shapes):
    """Predicts the body shape given extracted keypoints."""
    keypoints = np.array(keypoints).reshape(1, -1)  # Ensure correct shape (1, 99)
    prediction = model.predict(keypoints)
    predicted_class = np.argmax(prediction)  # Get class index
    return body_shapes[predicted_class]  # Return corresponding body shape label
