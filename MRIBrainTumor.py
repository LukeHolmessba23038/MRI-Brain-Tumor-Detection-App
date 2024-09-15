import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load your trained model with error handling
@st.cache_resource
def load_vgg_model():
    try:
        model = load_model('fine_tuned_vgg16_model.h5')  # Ensure the model file is correctly named and located
        st.write("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

model = load_vgg_model()

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
    if model:
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        return prediction[0][0]
    else:
        st.error("Model is not loaded.")
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

    # Predict the tumor
    if st.button("Classify"):
        with st.spinner("Classifying..."):
            prediction = predict_tumor(image)

        if prediction is not None:
            st.write(f"**Prediction Confidence: {prediction:.2f}**")

            # Set the threshold (adjust based on performance)
            threshold = 0.5  # You can experiment with different threshold values

            if prediction > threshold:
                st.write("**Prediction: Tumor Detected**")
            else:
                st.write("**Prediction: No Tumor Detected**")
        else:
            st.error("Prediction failed.")
