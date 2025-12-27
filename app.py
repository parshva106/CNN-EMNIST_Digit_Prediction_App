import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import pickle

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
st.set_page_config(page_title="Handwritten Character Recognition")

st.title("✍️ Draw a Digit OR Alphabet")
st.write("Draw inside the box. Use **white** pen on **black** background.")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)

predict_btn = st.button("Predict")

# -----------------------------
# Prediction
# -----------------------------
if predict_btn and canvas_result.image_data is not None:
    
    # Convert RGBA → grayscale PIL image
    img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")

    # Resize to 28×28
    img = img.resize((28, 28))

    # Invert colors (white writing on black bg)
    img = ImageOps.invert(img)

    # Convert to NumPy array
    img = np.array(img)

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
