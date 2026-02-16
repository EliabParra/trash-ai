import os
import tensorflowjs as tfjs
import tensorflow as tf

MODEL_PATH = os.path.join('models', 'trashnet_cnn_v1.keras')
OUTPUT_DIR = os.path.join('models', 'tfjs_model')

def export():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Loading model from {MODEL_PATH}...")
    # Load the trained model
    trained_model = tf.keras.models.load_model(MODEL_PATH)
    
    print("Reconstructing clean inference model...")
    # Rebuild architecture without GaussianNoise for clean export
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None # we will load from trained_model
    )
    
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    # Dropout is active only during training, but we can skip it for inference export if needed
    # x = tf.keras.layers.Dropout(0.2)(x) 
    outputs = tf.keras.layers.Dense(6, activation='softmax')(x)
    
    clean_model = tf.keras.Model(inputs, outputs)
    
    # Transfer weights
    # Note: This is tricky if layers don't match exactly by name/order.
    # An easier way is just to load the model and save it as a simple SavedModel first.
    # But since that failed, let's try just exporting the loaded model but forcing it to be a "serving" signature.
    
    print(f"Exporting to TensorFlow.js format at {OUTPUT_DIR}...")
    try:
        # Try converting via SavedModel again but with specific signature
        tfjs.converters.save_keras_model(trained_model, OUTPUT_DIR)
        print("Export completed successfully.")
    except Exception as e:
        print(f"Error exporting model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    export()
