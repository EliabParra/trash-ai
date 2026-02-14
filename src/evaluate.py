import os
import numpy as np
import warnings
# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import create_generators, IMG_height, IMG_width

MODEL_PATH = os.path.join('models', 'trashnet_cnn_v1.keras')
CONFUSION_MATRIX_PATH = 'confusion_matrix.png'

def evaluate(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train the model first.")
        return

    # 1. Load Data
    # Note: validation generator is shuffled=False by default in data_loader.py, which is crucial for evaluation
    _, val_gen = create_generators()
    
    # 2. Load Model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # 3. Evaluate
    print("Evaluating model...")
    results = model.evaluate(val_gen, verbose=1)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    
    # 4. Predictions for detailed metrics
    print("Generating predictions...")
    y_pred_prob = model.predict(val_gen)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = val_gen.classes
    class_labels = list(val_gen.class_indices.keys())
    
    # 5. Classification Report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_labels)
    print(report)
    
    # 6. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(CONFUSION_MATRIX_PATH)
    print(f"\nConfusion matrix saved to {os.path.abspath(CONFUSION_MATRIX_PATH)}")

if __name__ == "__main__":
    evaluate()
