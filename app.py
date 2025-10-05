# app.py

import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import json

# --- CONFIGURATION ---
MODEL_PATH = "model.onnx"  # Make sure this path is correct
# This JSON file will map the integer output of your model to a class name and provide extra info.
# We will create this file in the next step.
CLASS_INFO_PATH = "class_info.json" 
IMAGE_SIZE = 224  # The input size of your model (e.g., 224x224 pixels)

# --- HELPER FUNCTIONS ---

@st.cache_resource
def load_model(model_path):
    """Loads the ONNX model and returns an inference session."""
    try:
        session = ort.InferenceSession(model_path)
        st.success("✅ Model loaded successfully!")
        return session
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.error("Please ensure the 'model.onnx' file is in the same directory and is a valid ONNX model.")
        return None

@st.cache_data
def load_class_info(info_path):
    """Loads the class names and descriptions from a JSON file."""
    try:
        with open(info_path, 'r') as f:
            class_info = json.load(f)
        return class_info
    except FileNotFoundError:
        st.error(f"❌ Error: The file '{info_path}' was not found.")
        st.error("Please create this file with your model's class names and details.")
        return None

def preprocess_image(image_file):
    """Preprocesses the uploaded image to be model-ready."""
    try:
        # Open the image
        image = Image.open(image_file).convert('RGB')
        
        # Resize to the target size
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        
        # Convert to numpy array and normalize
        image_np = np.array(image, dtype=np.float32) / 255.0
        
        # Add a batch dimension
        image_np = np.expand_dims(image_np, axis=0)
        
        return image_np
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def predict(session, processed_image):
    """Runs inference and returns the prediction and confidence."""
    # Get the name of the input node
    input_name = session.get_inputs()[0].name
    
    # Run the model
    result = session.run(None, {input_name: processed_image})
    
    # Process the output
    probabilities = result[0][0] # Assuming the output is a batch of one
    predicted_class_index = np.argmax(probabilities)
    confidence = float(np.max(probabilities))
    
    return predicted_class_index, confidence

# --- STREAMLIT UI ---

st.set_page_config(page_title="AgriFutura AI Diagnosis", page_icon="🌿", layout="centered")

st.title("🌿 AgriFutura AI Diagnosis")
st.markdown("Upload an image of a plant leaf, and the AI will identify potential diseases.")

# Load the model and class info
model_session = load_model(MODEL_PATH)
class_info = load_class_info(CLASS_INFO_PATH)

# File uploader
uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_session is not None and class_info is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    st.info("🔍 Analyzing the image...")
    
    # Preprocess the image and make a prediction
    processed_image = preprocess_image(uploaded_file)
    
    if processed_image is not None:
        predicted_index, confidence = predict(model_session, processed_image)
        
        # Get the class info for the predicted index
        class_details = class_info.get(str(predicted_index))
        
        if class_details:
            disease_name = class_details["name"]
            advice = class_details["advice"]
            
            st.success(f"**Prediction:** {disease_name.replace('_', ' ')}")
            
            # Display confidence using a progress bar
            st.write("**Confidence Score:**")
            st.progress(confidence)
            st.write(f"{confidence:.2%}")
            
            # Display advice in an expander
            with st.expander("🔬 View Advice & Details"):
                st.write(advice)
        else:
            st.error(f"Could not find details for predicted class index: {predicted_index}")

# Add a sidebar for more info
st.sidebar.header("About")
st.sidebar.info(
    "This app uses a Deep Learning model to detect diseases in various plants. "
    "It's built with Streamlit and ONNX Runtime. "
    "Created by Aditya Goyal."
)
st.sidebar.header("How to Use")
st.sidebar.markdown(
    """
    1.  **Upload Image:** Click the 'Browse files' button.
    2.  **Select a File:** Choose a clear JPG, JPEG, or PNG image of a single plant leaf.
    3.  **Get Diagnosis:** The model will analyze the image and provide a diagnosis and advice.
    """
)