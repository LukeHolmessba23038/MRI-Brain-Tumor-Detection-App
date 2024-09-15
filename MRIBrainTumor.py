import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load your trained model
try:
    model = load_model('fine_tuned_vgg16_model.h5')  # Ensure your model file is in the same directory
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Define image size (ensure it matches the input size of your model)
IMAGE_SIZE = (128, 128)

# Function to preprocess the uploaded image
def preprocess_image(image):
    try:
        img = image.resize(IMAGE_SIZE)  # Resize the image to match the model's expected input
        img = img.convert('RGB')        # Ensure it's a 3-channel RGB image
        img = np.array(img) / 255.0     # Normalize pixel values to [0, 1]
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        return None

# Function to make predictions
def predict_tumor(image):
    processed_image = preprocess_image(image)
    if processed_image is not None:
        prediction = model.predict(processed_image)
        return prediction[0][0]
    else:
        return None

# Streamlit app
st.title("Brain MRI Tumor Detection")
st.write("Upload an MRI image to check for tumors.")

# Image uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Add a slider to adjust the threshold
    threshold = st.slider("Set Prediction Threshold", 0.0, 1.0, 0.5)

    # Predict the tumor
    if st.button("Classify"):
        with st.spinner("Classifying..."):
            prediction = predict_tumor(image)
        
        if prediction is not None:
            st.write(f"**Prediction Confidence: {prediction:.2f}**")
            if prediction > threshold:
                st.write("**Prediction: Tumor Detected**")
            else:
                st.write("**Prediction: No Tumor Detected**")
        else:
            st.error("Failed to make a prediction.")
