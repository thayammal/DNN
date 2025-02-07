import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("DL_model.h5")

# Streamlit UI
st.title("Handwritten Digit Recognition 🖊️🔢")
st.write("Upload an image of a handwritten digit (0-9) for prediction.")

# Upload file
uploaded_file = st.file_uploader(
    "Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = np.array(image.convert("L"))  # Convert to grayscale
    img = cv2.resize(img, (28, 28))  # Resize to 28x28 (like MNIST dataset)
    img = img / 255.0  # Normalize
    img = img.reshape(1, 28, 28, 1)  # Reshape for model input

    # Make prediction
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)

    # Display result
    st.success(f"Predicted Digit: {predicted_digit}")
