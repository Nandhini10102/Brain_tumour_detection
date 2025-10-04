import streamlit as st
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# -----------------------------
# Model Paths
# -----------------------------
MODEL_PATHS = {
    "Custom CNN": "models/custom_cnn.keras",
    "ResNet50 Transfer Learning": "models/resnet50_transfer.keras"
}

# Define tumor classes (same order as training)
class_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Dataset path (for showing examples)
DATASET_PATH = "data/Tumour/train"

# -----------------------------
# Preprocess Function
# -----------------------------
def preprocess_image(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (160, 160))  # must match training size
    img = img.astype("float32") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="🧠 Brain Tumor Classification", layout="wide")

st.title("🧠 Brain Tumor MRI Classification")
st.markdown("Upload an MRI scan and the model will predict the tumor type.")

# Sidebar - Settings
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Select Model", list(MODEL_PATHS.keys()))

# Load selected model
@st.cache_resource
def load_selected_model(choice):
    return load_model(MODEL_PATHS[choice])

model = load_selected_model(model_choice)

# Sidebar - Example Images
st.sidebar.header("📊 Example MRI Scans")

gallery_mode = st.sidebar.radio("Show examples as:", ["Single Sample", "Multiple Samples (Grid)"])

if os.path.exists(DATASET_PATH):
    for label in class_labels:
        folder = os.path.join(DATASET_PATH, label)
        if os.path.exists(folder):
            if gallery_mode == "Single Sample":
                # Show one random image
                sample_img = random.choice(os.listdir(folder))
                img_path = os.path.join(folder, sample_img)
                st.sidebar.image(img_path, caption=label, use_container_width=True)
            else:
                # Show up to 4 random images in a row
                sample_imgs = random.sample(os.listdir(folder), min(4, len(os.listdir(folder))))
                st.sidebar.markdown(f"**{label}**")
                cols = st.sidebar.columns(len(sample_imgs))
                for col, img_name in zip(cols, sample_imgs):
                    img_path = os.path.join(folder, img_name)
                    col.image(img_path, use_container_width=True)

# -----------------------------
# Upload Section
# -----------------------------
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded MRI", use_container_width=True)
    
    # Preprocess & Predict
    img = preprocess_image(uploaded_file)
    preds = model.predict(img)[0]
    
    # Results
    pred_idx = np.argmax(preds)
    pred_label = class_labels[pred_idx]
    confidence = preds[pred_idx] * 100

    st.subheader(f"Prediction using **{model_choice}**: {pred_label} 🩺")
    st.write(f"Confidence: **{confidence:.2f}%**")

    # Confidence Scores Chart
    st.markdown("### Confidence Scores by Class:")
    fig, ax = plt.subplots(figsize=(6,3))
    ax.barh(class_labels, preds * 100, color="skyblue")
    ax.set_xlim([0, 100])
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Prediction Confidence per Class")
    st.pyplot(fig)



