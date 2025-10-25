import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os
import gdown

# -------------------------------
# Streamlit App Config
# -------------------------------
st.set_page_config(page_title="MNIST Digit Recognition", layout="centered")
st.title("🧠 MNIST Digit Recognition App")

st.write("Upload a handwritten **digit (0–9)** image and the model will predict it!")

# -------------------------------
# Load Trained Model from Google Drive
# -------------------------------
model_path = "mnist_cnn.h5"

# Download model if it doesn't exist
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?export=download&id=1NF-w35UmAzC_Q8_ln5CDhXSt9NLj-y83"
    st.info("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

# Load model
model = load_model(model_path)
st.success("Model loaded successfully ✅")

# -------------------------------
# Upload Image
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Step 1: Open and preprocess image
        image = Image.open(uploaded_file).convert('L')   # convert to grayscale
        image = ImageOps.invert(image)                   # invert for white text on black bg
        image = image.resize((28, 28))                   # resize to 28x28
        
        st.image(image, caption="🖼 Uploaded Image", width=150)
        
        # Step 2: Prepare array
        img_array = np.array(image).reshape(1, 28, 28, 1).astype('float32') / 255.0
        
        # Step 3: Predict
        with st.spinner("Predicting..."):
            prediction = model.predict(img_array)
            predicted_digit = np.argmax(prediction)
        
        st.success(f"✅ Predicted Digit: **{predicted_digit}**")

        # Step 4: Show probabilities
        st.subheader("📊 Prediction Probabilities")
        fig, ax = plt.subplots()
        ax.bar(range(10), prediction[0])
        ax.set_xticks(range(10))
        ax.set_xlabel("Digit")
        ax.set_ylabel("Probability")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")
else:
    st.info("Please upload a digit image (JPG/PNG).")

