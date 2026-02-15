import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, GaussianNoise, Dropout
from tensorflow.keras.models import Model

def build_model(input_shape=(224, 224, 3), num_classes=6):
    """
    Builds the MobileNetV2 Transfer Learning model for TrashNet.
    Optimized for Accuracy > 90% and efficient execution.
    """
    # 1. Base Model: MobileNetV2 pre-trained on ImageNet
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model weights initially
    base_model.trainable = False
    
    # 2. Add custom classification head
    inputs = Input(shape=input_shape)
    
    # FocalX: Robustness layer
    x = GaussianNoise(0.01)(inputs) 
    
    # MobileNetV2 layers
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x) # Keeping some dropout for regularization
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model, base_model

if __name__ == "__main__":
    model = build_model()
    model.summary()
