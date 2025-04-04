import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# 1️⃣ Load Dataset (Using Keypoints Instead of Images)
def load_dataset():
    num_samples = 1000
    num_keypoints = 99  # Adjust based on MediaPipe keypoints
    num_classes = 5

    X = np.random.rand(num_samples, num_keypoints)  # Simulated keypoints
    y = np.random.randint(0, num_classes, num_samples)

    X = X / np.max(X)  # Normalize keypoints
    y = keras.utils.to_categorical(y, num_classes)

    return X, y, num_classes

# 2️⃣ Load Data
X, y, num_classes = load_dataset()

# 3️⃣ Split Data (80% Train, 20% Test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Build Model (Fully Connected Instead of CNN)
def build_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# 5️⃣ Train the Model
model = build_model(X_train.shape[1], num_classes)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# 6️⃣ Save Model
model.save("body_detection_model.h5")
print("✅ Model trained and saved successfully!")
