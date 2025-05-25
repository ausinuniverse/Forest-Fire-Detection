import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def get_class_names(data_dir):
    """Get class names from directory structure"""
    return sorted([item.name for item in Path(data_dir).glob('*') if item.is_dir()])

def load_and_validate_dataset(data_dir, img_size=(224, 224), batch_size=32, 
                           validation_split=None, subset=None, shuffle=True):
    """Create dataset with robust validation"""
    # First count valid files
    valid_files = []
    for class_name in get_class_names(data_dir):
        class_dir = Path(data_dir) / class_name
        for img_file in class_dir.glob('*'):
            try:
                img = tf.io.read_file(str(img_file))
                tf.image.decode_jpeg(img, channels=3)
                valid_files.append(str(img_file))
            except:
                print(f"Skipping corrupt file: {img_file}")
                continue
    
    # Create dataset from valid files
    if not valid_files:
        raise ValueError(f"No valid images found in {data_dir}")
    
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset=subset,
        seed=123,
        shuffle=shuffle,
        batch_size=batch_size,
        image_size=img_size,
        label_mode='binary'
    )
    
    return ds

def prepare_dataset(ds, augment=False):
    """Prepare dataset with validation"""
    # Filter out any potential empty batches
    ds = ds.filter(lambda x, y: tf.shape(x)[0] > 0)
    
    normalization_layer = layers.Rescaling(1./255)
    ds = ds.map(lambda x, y: (normalization_layer(x), y))
    
    if augment:
        augmentation_layers = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
        ds = ds.map(lambda x, y: (augmentation_layers(x, training=True), y))
    
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

def safe_visualize_dataset(ds, class_names, rows=3, cols=3):
    """Safe visualization with error handling"""
    plt.figure(figsize=(10, 10))
    try:
        # Take one batch safely
        for batch in ds.take(1):
            images, labels = batch
            if len(images) == 0:
                print("Warning: Empty batch encountered")
                return
            
            # Display up to rows*cols images
            for i in range(min(rows*cols, len(images))):
                ax = plt.subplot(rows, cols, i+1)
                plt.imshow(images[i].numpy().astype("float32"))
                plt.title(class_names[int(labels[i])])
                plt.axis("off")
        plt.show()
    except Exception as e:
        print(f"Visualization error: {str(e)}")