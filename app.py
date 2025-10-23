import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load trained model
model = load_model("cifar10_cnn_model.h5")

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

st.title("CIFAR-10 Image Classifier")
st.write("Upload an image (any of the 10 CIFAR-10 categories) and the model will predict it.")
st.write("The 10 categories are: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((32,32))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    st.write(f"### Predicted Class: {class_names[pred_class]}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")
