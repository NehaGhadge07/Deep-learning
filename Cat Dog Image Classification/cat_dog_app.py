import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("cat_dog_model.h5")

# Streamlit app title
st.title("🐱🐶 Cat vs Dog Classifier")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(img_array)
    class_name = "🐶 Dog" if prediction[0][0] > 0.5 else "🐱 Cat"
    
    # Display prediction
    st.write(f"### Prediction: {class_name}")
