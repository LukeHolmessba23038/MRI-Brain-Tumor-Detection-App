import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Streamlit app title
st.title("Brain MRI Tumor Detection")

# Model file uploader (this will allow you to upload your model in case Streamlit can't access Git LFS)
model_file = st.file_uploader("Upload the model file (.h5)", type=["h5"])

if model_file is not None:
    # Load the uploaded model file
    model = load_model(model_file)
    
    # Define image size (ensure it matches the input size of your model)
    IMAGE_SIZE = (128, 128)

    # Function to preprocess the uploaded image
    def preprocess_image(image):
        img = image.resize(IMAGE_SIZE)  # Resize the image to match the model's expected input
        img = img.convert('RGB')        # Ensure it's a 3-channel RGB image
        img = np.array(img) / 255.0     # Normalize pixel values to [0, 1]
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img

    # Function to make predictions
    def predict_tumor(image):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        return prediction[0][0]

    # Image uploader
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image", use_column_width=True)

        # Predict the tumor
        if st.button("Classify"):
            with st.spinner("Classifying..."):
                prediction = predict_tumor(image)

            # Set the threshold (adjust based on performance)
            threshold = 0.5  # You can experiment with different threshold values

            if prediction > threshold:
                st.write("**Prediction: Tumor Detected**")
            else:
                st.write("**Prediction: No Tumor Detected**")
