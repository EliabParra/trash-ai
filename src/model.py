import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.layers import GaussianNoise

def build_model(input_shape=(64, 64, 3), num_classes=6):
    """
    Builds the CNN model architecture for TrashNet classification.
    Includes FocalX-inspired robustness features (GaussianNoise).
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels).
        num_classes (int): Number of output classes.
        
    Returns:
        tf.keras.Model: The compiled CNN model (uncompiled, just architecture).
    """
    model = Sequential([
        Input(shape=input_shape),
        
        # FocalX Robustness Layer: Gaussian Noise to simulate sensor noise/adversarial perturbations
        # This acts as a smoothing mechanism to improve generalization and robustness.
        GaussianNoise(0.1), 
        
        # Block 1
        Conv2D(32, (2, 2), activation='relu', padding='same', strides=1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        
        # Block 2
        Conv2D(32, (2, 2), activation='relu', padding='same', strides=1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        
        # Block 3
        Conv2D(32, (2, 2), activation='relu', padding='same', strides=1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        
        # Classification Head
        Flatten(),
        Dense(512, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()
