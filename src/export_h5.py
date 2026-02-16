import os
import tensorflow as tf

MODEL_PATH = os.path.join('models', 'trashnet_cnn_v1.keras')
OUTPUT_PATH = os.path.join('models', 'model.h5')

def export_h5():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Loading model from {MODEL_PATH}...")
    # Load original model
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print(f"Saving model to {OUTPUT_PATH} (H5 format)...")
    # Save as .h5
    model.save(OUTPUT_PATH, save_format='h5')
    print("Export completed successfully.")

if __name__ == "__main__":
    export_h5()
