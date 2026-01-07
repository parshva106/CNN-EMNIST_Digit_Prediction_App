import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import pickle


# -----------------------------
# Load Model & Labels
# -----------------------------
@st.cache_resource
def load_cnn_model():
    return load_model("emnist_cnn_model.h5")


@st.cache_resource
def load_labels():
    with open("label_map.pkl", "rb") as f:
        return pickle.load(f)


model = load_cnn_model()
LABELS = load_labels()


# -----------------------------
# UI
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

    img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
    img = ImageOps.invert(img)
    img = ImageOps.autocontrast(img)
    img = img.resize((28, 28))

    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)

    pred = model.predict(img)

    idx = int(np.argmax(pred))
    prob = float(np.max(pred))
    char = LABELS[idx]

    st.write("🔍 DEBUG INFO")
    st.write("Index:", idx)
    st.write("Label from LABELS:", char)
    st.write("First 20 labels:", LABELS[:20])
    st.write("Total labels:", len(LABELS))

    st.subheader(f"Prediction: **{char}**")
    st.write(f"Confidence: `{prob*100:.2f}%`")
