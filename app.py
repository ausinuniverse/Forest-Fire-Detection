# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
from pathlib import Path
import os
import time

# Configuration
MODELS_DIR = Path('models')
THRESHOLD = 0.5  # Confidence threshold for fire detection

# Streamlit page config
st.set_page_config(
    page_title="Forest Fire Detector",
    page_icon="🔥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .prediction-card {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .fire {
        background-color: #ffcccc;
        border-left: 5px solid #ff0000;
    }
    .no-fire {
        background-color: #ccffcc;
        border-left: 5px solid #00aa00;
    }
    .plot-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_metadata():
    """Load the latest model and metadata"""
    # Find metadata
    metadata_files = list(MODELS_DIR.glob('metadata.json'))
    if not metadata_files:
        st.error("No metadata.json found in models directory")
        return None, None
    
    # Load metadata
    with open(metadata_files[0], 'r') as f:
        metadata = json.load(f)
    
    # Find the latest model
    model_files = [f for f in MODELS_DIR.glob('*.h5') if 'best_model' not in f.name]
    if not model_files:
        st.error("No model files found in models directory")
        return None, None
    
    latest_model = sorted(model_files, key=lambda x: x.name)[-1]
    
    # Load model with progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Loading model...")
    model = tf.keras.models.load_model(latest_model)
    
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    
    status_text.text("Model loaded successfully!")
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    
    return model, metadata

def preprocess_image(img, target_size):
    """Preprocess image for model prediction"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

def create_prediction_plot(img, prediction, confidence, class_names):
    """Create matplotlib plot for visualization"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    
    # Customize based on prediction
    if prediction == class_names[1]:  # Fire
        title_color = 'red'
        box_color = 'lightcoral'
    else:
        title_color = 'green'
        box_color = 'lightgreen'
    
    title = f"Prediction: {prediction}\nConfidence: {confidence:.2%}"
    ax.set_title(title, fontsize=16, color=title_color, pad=20)
    ax.axis('off')
    
    # Add colored border
    for spine in ax.spines.values():
        spine.set_edgecolor(title_color)
        spine.set_linewidth(8)
    
    # Add prediction info box
    textstr = f"""
    Model: EfficientNetB0
    Threshold: {THRESHOLD:.0%}
    Confidence: {confidence:.2%}
    """
    props = dict(boxstyle='round', facecolor=box_color, alpha=0.5)
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, 
            fontsize=12, verticalalignment='bottom', bbox=props)
    
    return fig

def main():
    st.title("🔥 Forest Fire Detection System")
    st.markdown("""
    Upload images to detect whether they contain forest fires using our deep learning model.
    The system uses EfficientNetB0 architecture trained on thousands of fire/non-fire images.
    """)
    
    # Load model
    model, metadata = load_model_and_metadata()
    if model is None or metadata is None:
        return
    
    class_names = metadata['class_names']
    img_size = tuple(metadata['img_size'])
    
    # Sidebar controls
    st.sidebar.header("Settings")
    global THRESHOLD
    THRESHOLD = st.sidebar.slider(
        "Detection Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Adjust the sensitivity of fire detection"
    )
    
    st.sidebar.markdown("### Model Info")
    st.sidebar.write(f"**Architecture:** EfficientNetB0")
    st.sidebar.write(f"**Input Size:** {img_size[0]}×{img_size[1]} pixels")
    st.sidebar.write(f"**Classes:** {', '.join(class_names)}")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose images to analyze",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )
    
    if not uploaded_files:
        st.info("Please upload one or more images to analyze")
        return
    
    # Process each image
    for uploaded_file in uploaded_files:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Uploaded Image")
            img = Image.open(uploaded_file)
            st.image(img, caption=uploaded_file.name, use_column_width=True)
        
        with col2:
            st.subheader("Analysis Results")
            
            # Preprocess and predict
            img_array = preprocess_image(img, img_size)
            prediction = model.predict(img_array, verbose=0)
            confidence = float(prediction[0][0])
            pred_class = class_names[1] if confidence >= THRESHOLD else class_names[0]
            
            # Display prediction card
            card_class = "fire" if pred_class == class_names[1] else "no-fire"
            st.markdown(
                f"""
                <div class="prediction-card {card_class}">
                    <h3 style="margin-top:0;">Prediction: {pred_class}</h3>
                    <p><b>Confidence:</b> {confidence:.2%}</p>
                    <p><b>Threshold:</b> {THRESHOLD:.0%}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Create and display plot
            fig = create_prediction_plot(img, pred_class, confidence, class_names)
            with st.expander("Detailed Analysis", expanded=True):
                st.pyplot(fig)
            
            # Add divider between images
            st.markdown("---")

if __name__ == "__main__":
    main()