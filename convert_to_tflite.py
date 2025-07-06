import tensorflow as tf

def convert_model_to_tflite(h5_model_path="body_detection_model.h5", tflite_model_path="body_detection_model.tflite"):
    # Load your Keras model
    model = tf.keras.models.load_model(h5_model_path)

    # Create the converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optional: Enable quantization for optimization (uncomment to enable)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Convert the model
    tflite_model = converter.convert()

    # Save the converted model
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    print(f"✅ Model converted and saved to {tflite_model_path}")

if __name__ == "__main__":
    convert_model_to_tflite()
