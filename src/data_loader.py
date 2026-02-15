import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import shutil

# Constants
RAW_DATA_DIR = os.path.join('data', 'raw', 'dataset-resized')
IMG_height = 224
IMG_width = 224
BATCH_SIZE = 32
RANDOM_SEED = 42

def create_generators():
    """
    Creates training and validation data generators with 75/25 split.
    Uses 'validation_split' feature of ImageDataGenerator for efficiency.
    """
    
    # 1. Generator for Training (with Augmentation)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.25
    )

    # 2. Generator for Validation (No Augmentation, only Rescale)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.25
    )

    print(f"Loading data from: {os.path.abspath(RAW_DATA_DIR)}")

    train_generator = train_datagen.flow_from_directory(
        RAW_DATA_DIR,
        target_size=(IMG_height, IMG_width),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        seed=RANDOM_SEED,
        shuffle=True
    )

    validation_generator = val_datagen.flow_from_directory(
        RAW_DATA_DIR,
        target_size=(IMG_height, IMG_width),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        seed=RANDOM_SEED,
        shuffle=False
    )

    return train_generator, validation_generator

if __name__ == "__main__":
    train_gen, val_gen = create_generators()
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Classes: {list(train_gen.class_indices.keys())}")
