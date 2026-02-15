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
    model, base_model = build_model(input_shape=(IMG_height, IMG_width, 3), num_classes=num_classes)
    
    # 3. Compile Model (Stage 1: Training only the head)
    print("Stage 1: Training the custom classification head...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 4. Define Callbacks
    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODEL_DIR, MODEL_NAME)
    
    callbacks_s1 = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    # 5. Train Stage 1
    model.fit(
        train_gen,
        epochs=30, # Sufficient for initial convergence
        validation_data=val_gen,
        callbacks=callbacks_s1,
        verbose=1
    )
    
    # 6. Stage 2: Fine-Tuning
    print("\nStage 2: Fine-tuning the base model...")
    # Unfreeze the base model
    base_model.trainable = True
    
    # Fine-tune from this layer onwards (optional, but unfreezing everything works too)
    # We use a very low learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks_s2 = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    # Train Stage 2
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks_s2,
        verbose=1
    )
    
    print(f"Fine-tuning completed. Best model saved to {checkpoint_path}")
    
    return history

if __name__ == "__main__":
    train()
