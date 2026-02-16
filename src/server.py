"""
TrashAI — Single Server
Flask serves static files + inference API.
Model loaded once at startup = fast predictions.
"""
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf

# ── Config ──
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'trashnet_cnn_v1.keras')
STATIC_DIR = os.path.join(os.path.dirname(__file__), '..', 'web', 'public')
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
CLASS_INFO = {
    'cardboard': {'emoji': '📦', 'name': 'Cartón',         'color': '#A0522D', 'tip': 'Aplánalo antes de reciclarlo para ahorrar espacio.'},
    'glass':     {'emoji': '🍶', 'name': 'Vidrio',         'color': '#4FC3F7', 'tip': 'Enjuágalo y deposítalo en el contenedor verde.'},
    'metal':     {'emoji': '🥫', 'name': 'Metal',          'color': '#78909C', 'tip': 'Latas y aluminio van al contenedor amarillo.'},
    'paper':     {'emoji': '📄', 'name': 'Papel',          'color': '#FFF176', 'tip': 'No mezcles papel mojado o con grasa.'},
    'plastic':   {'emoji': '🧴', 'name': 'Plástico',       'color': '#EF5350', 'tip': 'Revisa el número de reciclaje en la base.'},
    'trash':     {'emoji': '🗑️', 'name': 'Basura General', 'color': '#9E9E9E', 'tip': 'Este residuo no es reciclable, va al contenedor gris.'}
}

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='')

# ── Load Model Once ──
print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded and ready!")

# ── Serve Frontend ──
@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')

# ── Prediction API ──
@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400

    file = request.files['image']
    tmp_path = '/tmp/trashai_upload'
    file.save(tmp_path)

    try:
        # Preprocess: resize to 224x224, normalize to [0, 1]
        img = tf.keras.utils.load_img(tmp_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # Predict (model already in memory = fast)
        predictions = model.predict(img_batch, verbose=0)
        probabilities = predictions[0]

        # Build results
        results = []
        for i, cls in enumerate(CLASSES):
            info = CLASS_INFO[cls]
            results.append({
                'class': cls,
                'probability': round(float(probabilities[i]) * 100, 2),
                **info
            })
        results.sort(key=lambda x: x['probability'], reverse=True)

        return jsonify({
            'success': True,
            'prediction': results[0],
            'allResults': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ── Health Check ──
@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

if __name__ == '__main__':
    print("\n🚀 TrashAI running at http://localhost:3000\n")
    app.run(host='0.0.0.0', port=3000)
