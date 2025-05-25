# train_and_save_model.py
import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks
from pathlib import Path
import utils.dataset_utils as ds_utils
import matplotlib.pyplot as plt
import warnings
from urllib3.exceptions import NotOpenSSLWarning
import os
import json

# Suppress SSL warnings
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

# Configuration
DATA_DIR = Path('data/Train_Data')
TEST_DIR = Path('data/Test_Data')
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30

def save_model_with_metadata(model, class_names, img_size):
    """Save model along with its metadata"""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    model.save('models/forest_fire_model.h5')
    
    # Save metadata
    metadata = {
        'class_names': class_names,
        'img_size': img_size
    }
    
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f)
        
    print("Model and metadata saved successfully")

def main():
    try:
        # Verify directories
        if not DATA_DIR.exists() or not TEST_DIR.exists():
            raise FileNotFoundError("Data directories not found")
        
        # Get class names
        class_names = ds_utils.get_class_names(DATA_DIR)
        print(f"Class names: {class_names}")
        
        # Load datasets
        print("Loading training data...")
        train_ds = ds_utils.load_and_validate_dataset(
            DATA_DIR, IMG_SIZE, BATCH_SIZE, 
            validation_split=0.2, subset='training'
        )
        
        print("Loading validation data...")
        val_ds = ds_utils.load_and_validate_dataset(
            DATA_DIR, IMG_SIZE, BATCH_SIZE,
            validation_split=0.2, subset='validation'
        )
        
        print("Loading test data...")
        test_ds = ds_utils.load_and_validate_dataset(
            TEST_DIR, IMG_SIZE, BATCH_SIZE,
            shuffle=False
        )
        
        # Prepare datasets
        print("Preparing datasets...")
        train_ds = ds_utils.prepare_dataset(train_ds, augment=True)
        val_ds = ds_utils.prepare_dataset(val_ds)
        test_ds = ds_utils.prepare_dataset(test_ds)
        
        # Try visualization
        print("Attempting visualization...")
        ds_utils.safe_visualize_dataset(train_ds, class_names)
        
        # Model definition
        print("Creating model...")
        base_model = applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=IMG_SIZE + (3,)
        )
        base_model.trainable = False
        
        inputs = layers.Input(shape=IMG_SIZE + (3,))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = models.Model(inputs, outputs)
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        # Callbacks
        callbacks = [
            callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            callbacks.ModelCheckpoint('models/best_model.h5', save_best_only=True),
            callbacks.CSVLogger('training_log.csv')
        ]
        
        # Train model
        print("Starting training...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks
        )
        
        # Save model with metadata
        save_model_with_metadata(model, class_names, IMG_SIZE)
        
        # Plot training history
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
        plt.savefig('training_history.png')
        plt.show()
        
    except Exception as e:
        print(f"\nError encountered: {str(e)}")
        print("Please check your dataset and try again.")
        raise

if __name__ == "__main__":
    main()