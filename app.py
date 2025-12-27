import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import pickle
import cv2

# -----------------------------
# Load model & labels
# -----------------------------
@st.cache_resource
def load_cnn_model():
    return load_model("emnist_cnn_model.h5")

@st.cache_resource
def load_labels():
    with open("label_map.pkl", "rb") as f:
        return pickle.load(f)

model = load_cnn_model()
label_map = load_labels()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Handwritten Character Recognition", layout="centered")

st.title("✍️ Draw a Digit or Alphabet")
st.write("Draw clearly in the box below. Use white pen on black background.")

# Canvas settings
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=300,
    height=300,
    drawing_mode="freedraw",
    key="canvas",
)

predict_btn = st.button("Predict")

# -----------------------------
# Prediction
# -----------------------------
if predict_btn:
    if canvas_result.image_data is not None:
        # Convert to grayscale image
        img = canvas_result.image_data.astype("uint8")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize to 28x28
        img = cv2.resize(img, (28, 28))

        # Invert (white digits on black bg)
        img = 255 - img

        # Normalize
        img = img / 255.0

        # Reshape for CNN
        img = img.reshape(1, 28, 28, 1)

        # Predict
        pred = model.predict(img)
        prob = np.max(pred)
        idx = np.argmax(pred)
        char = label_map[idx]

        st.subheader(f"Prediction: **{char}**")
        st.write(f"Confidence: `{prob*100:.2f}%`")
