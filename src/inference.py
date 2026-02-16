"""
TrashAI Inference Script
Takes an image path as argument, runs model prediction,
and prints results as JSON to stdout.
"""
import sys
import os
import json
import numpy as np

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'trashnet_cnn_v1.keras')
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Load model once (cached via module-level variable)
model = None

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

def predict(image_path):
    model = load_model()
    
    # Load and preprocess image
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_batch, verbose=0)
    probabilities = predictions[0]
    
    # Build results sorted by probability
    results = []
    for i, cls in enumerate(CLASSES):
        results.append({
            'class': cls,
            'probability': round(float(probabilities[i]) * 100, 2)
        })
    
    results.sort(key=lambda x: x['probability'], reverse=True)
    return results

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'No image path provided'}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(json.dumps({'error': f'Image not found: {image_path}'}))
        sys.exit(1)
    
    results = predict(image_path)
    print(json.dumps(results))
