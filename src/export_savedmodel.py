import os
import tensorflow as tf

MODEL_PATH = os.path.join('models', 'trashnet_cnn_v1.keras')
OUTPUT_DIR = os.path.join('models', 'saved_model')

def export_savedmodel():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print(f"Saving as TensorFlow SavedModel to {OUTPUT_DIR}...")
    model.export(OUTPUT_DIR)
    print("SavedModel export completed successfully.")

if __name__ == "__main__":
    export_savedmodel()
