import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load trained model
model = load_model(r"E:\MNIST_cnn_Project\mnist_cnn.h5")

st.title("MNIST Digit Recognition")
st.write("Upload an image of a digit (0-9) below:")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    # Open and process image
    image = Image.open(uploaded_file).convert('L')  # grayscale
    image = ImageOps.invert(image)                 # invert if needed
    image = image.resize((28,28))
    
    st.image(image, caption='Uploaded Image', use_column_width=False)
    
    # Convert to array
    img_array = np.array(image).reshape(1,28,28,1)/255.0
    
    # Predict
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    
    st.success(f"Predicted Digit: {predicted_digit}")

