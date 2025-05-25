# predict_batch_fire.py

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
import csv
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
MODELS_DIR = Path('models')
INPUT_DIR = Path('new-images')  # Folder containing new images to predict
OUTPUT_DIR = Path('predictions')
THRESHOLD = 0.5  # Confidence threshold for fire detection
OUTPUT_CSV = OUTPUT_DIR / f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

def setup_directories():
    """Create necessary directories if they don't exist"""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_model_and_metadata():
    """Load the latest model and metadata from the models directory"""
    # Find the latest metadata file
    metadata_files = list(MODELS_DIR.glob('metadata.json'))
    if not metadata_files:
        raise FileNotFoundError("No metadata.json found in models directory")
    
    # Load metadata
    with open(metadata_files[0], 'r') as f:
        metadata = json.load(f)
    
    # Find the latest model file (excluding best_model files)
    model_files = [f for f in MODELS_DIR.glob('*.h5') if 'best_model' not in f.name]
    if not model_files:
        raise FileNotFoundError("No model files found in models directory")
    
    # Get the most recent model by timestamp
    latest_model = sorted(model_files, key=lambda x: x.name)[-1]
    
    # Load model
    model = tf.keras.models.load_model(latest_model)
    
    return model, metadata

def preprocess_image(img_path, target_size):
    """Load and preprocess an image for prediction"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

def predict_image(model, img_array, class_names, threshold=0.5):
    """Make prediction on a preprocessed image"""
    prediction = model.predict(img_array, verbose=0)
    confidence = float(prediction[0][0])
    
    if confidence >= threshold:
        predicted_class = class_names[1]  # Assuming class_names[1] is "Fire"
    else:
        predicted_class = class_names[0]  # Assuming class_names[0] is "No_Fire"
    
    return predicted_class, confidence

def save_prediction_to_csv(image_name, prediction, confidence, csv_path):
    """Save prediction results to CSV file"""
    file_exists = csv_path.exists()
    
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['image_name', 'prediction', 'confidence', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'image_name': image_name,
            'prediction': prediction,
            'confidence': f"{confidence:.4f}",
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

def save_prediction_plot(img_path, prediction, confidence, output_dir):
    """Save prediction visualization as image"""
    img = image.load_img(img_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f"Prediction: {prediction}\nConfidence: {confidence:.2%}")
    plt.axis('off')
    
    # Create output filename
    img_name = Path(img_path).stem
    output_path = output_dir / f"{img_name}_prediction.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def process_images(model, metadata):
    """Process all images in the input directory"""
    class_names = metadata['class_names']
    img_size = tuple(metadata['img_size'])
    
    # Get all image files in input directory
    image_files = list(INPUT_DIR.glob('*'))
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    if not image_files:
        print(f"No images found in {INPUT_DIR}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    for img_path in image_files:
        try:
            print(f"\nProcessing {img_path.name}...")
            
            # Preprocess and predict
            img_array = preprocess_image(img_path, img_size)
            prediction, confidence = predict_image(model, img_array, class_names, THRESHOLD)
            
            # Display results in console
            print(f"- Prediction: {prediction}")
            print(f"- Confidence: {confidence:.2%}")
            
            # Save results
            save_prediction_to_csv(img_path.name, prediction, confidence, OUTPUT_CSV)
            save_prediction_plot(img_path, prediction, confidence, OUTPUT_DIR)
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
            continue
    
    print(f"\nProcessing complete. Results saved to {OUTPUT_DIR}")

def main():
    try:
        setup_directories()
        
        print("Loading model and metadata...")
        model, metadata = load_model_and_metadata()
        
        print(f"\nModel loaded successfully")
        print(f"Class names: {metadata['class_names']}")
        print(f"Image size: {metadata['img_size']}")
        print(f"Threshold: {THRESHOLD:.0%}")
        
        process_images(model, metadata)
        
    except Exception as e:
        print(f"\n[ERROR]: {str(e)}")
        print("Prediction failed. Please check the model files and input directory.")

if __name__ == "__main__":
    main()