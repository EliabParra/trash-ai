import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.data_loader import create_generators, IMG_height, IMG_width
from src.model import build_model

# Constants
EPOCHS = 70
MODEL_DIR = 'models'
MODEL_NAME = 'trashnet_cnn_v1.keras' # Using .keras format as per new Keras standard

def train():
    # 1. Load Data
    train_gen, val_gen = create_generators()
    num_classes = len(train_gen.class_indices)
    
    # 2. Build Model
    model = build_model(input_shape=(IMG_height, IMG_width, 3), num_classes=num_classes)
    
    # 3. Compile Model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 4. Define Callbacks
    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODEL_DIR, MODEL_NAME)
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy', # Monitor validation accuracy to save best model
        save_best_only=True,
        verbose=1
    )
    
    # 5. Train Model
    print("Starting training...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    print(f"Training completed. Best model saved to {checkpoint_path}")
    
    return history

if __name__ == "__main__":
    train()
