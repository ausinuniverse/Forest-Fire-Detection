# train_and_save_model.py

import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks as tf_callbacks
from pathlib import Path
import utils.dataset_utils as ds_utils
import matplotlib.pyplot as plt
import warnings
from urllib3.exceptions import NotOpenSSLWarning
import os
import json
from datetime import datetime

# Suppress SSL warnings
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

# Configuration
DATA_DIR = Path('data/Train_Data')
TEST_DIR = Path('data/Test_Data')
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30

# Paths
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
FINAL_MODEL_PATH = MODELS_DIR / f'forest_fire_model_{TIMESTAMP}.h5'
BEST_MODEL_PATH = MODELS_DIR / f'best_model_{TIMESTAMP}.h5'
METADATA_PATH = MODELS_DIR / 'metadata.json'

def save_metadata(class_names, img_size):
    """Save metadata only"""
    metadata = {
        'class_names': class_names,
        'img_size': img_size
    }
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f)
    print(f"Metadata saved to {METADATA_PATH}")

def build_model(img_size):
    """Builds and returns the CNN model"""
    base_model = applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=img_size + (3,)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=img_size + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

def plot_history(history, output_path='training_history.png'):
    """Plot training accuracy and loss"""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def main():
    try:
        # Verify data directories
        if not DATA_DIR.exists() or not TEST_DIR.exists():
            raise FileNotFoundError("One or both data directories not found.")

        # Load class names
        class_names = ds_utils.get_class_names(DATA_DIR)
        print(f"Classes: {class_names}")

        # Load datasets
        print("Loading datasets...")
        train_ds = ds_utils.load_and_validate_dataset(DATA_DIR, IMG_SIZE, BATCH_SIZE, validation_split=0.2, subset='training')
        val_ds = ds_utils.load_and_validate_dataset(DATA_DIR, IMG_SIZE, BATCH_SIZE, validation_split=0.2, subset='validation')
        test_ds = ds_utils.load_and_validate_dataset(TEST_DIR, IMG_SIZE, BATCH_SIZE, shuffle=False)

        # Prepare datasets
        print("Preparing datasets...")
        train_ds = ds_utils.prepare_dataset(train_ds, augment=True)
        val_ds = ds_utils.prepare_dataset(val_ds)
        test_ds = ds_utils.prepare_dataset(test_ds)

        # Visualize dataset
        ds_utils.safe_visualize_dataset(train_ds, class_names)

        # Build and compile model
        print("Building model...")
        model = build_model(IMG_SIZE)

        # Callbacks
        cb_list = [
            tf_callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf_callbacks.ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True),
            tf_callbacks.CSVLogger('training_log.csv')
        ]

        # Train the model
        print("Training model...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=cb_list
        )

        # Save final trained model (last epoch, not necessarily best)
        model.save(FINAL_MODEL_PATH)
        print(f"Final model saved to {FINAL_MODEL_PATH}")

        # Save metadata
        save_metadata(class_names, IMG_SIZE)

        # Plot training history
        plot_history(history)

    except Exception as e:
        print(f"\n[ERROR]: {str(e)}")
        print("Training aborted. Please check paths, data, or configuration.")
        raise

if __name__ == "__main__":
    main()
