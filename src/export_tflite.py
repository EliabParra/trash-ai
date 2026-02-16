import os
import tensorflow as tf

MODEL_PATH = os.path.join('models', 'trashnet_cnn_v1.keras')
OUTPUT_PATH = os.path.join('models', 'trashnet_model.tflite')

def export():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print("Converting to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Enable optimizations (quantization) for smaller size
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    print(f"Saving TFLite model to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'wb') as f:
        f.write(tflite_model)
    
    print("Export completed successfully.")

if __name__ == "__main__":
    export()
