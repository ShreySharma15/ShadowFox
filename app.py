import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


model = tf.keras.models.load_model("cifar10_cnn_model.h5")

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# Streamlit UI

st.title("ShadowFox - CIFAR-10 Image Classifier")
st.write("Upload an image and let the model predict its class!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open the image and ensure it has 3 color channels
        image = Image.open(uploaded_file).convert("RGB")
        # Resize to 32x32 as required by CIFAR-10 model
        image_resized = image.resize((32, 32))
        
        # Display uploaded image
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess for model
        img_array = np.array(image_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict
        preds = model.predict(img_array)
        predicted_class = class_names[np.argmax(preds)]
        confidence = np.max(preds) * 100

        # Show results
        st.success(f"Prediction: **{predicted_class}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

    except Exception as e:
        st.error(f"Error processing the image: {e}")
