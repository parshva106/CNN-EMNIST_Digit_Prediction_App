import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps


# -----------------------------
# Load Model (cached)
# -----------------------------
@st.cache_resource
def load_cnn_model():
    return load_model("emnist_cnn_model.h5")


model = load_cnn_model()


# -----------------------------
# Label Map
# (Digits + Capital Letters)
# -----------------------------
LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


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

    # Invert → white text on black background
    img = ImageOps.invert(img)

    # Improve contrast (helps recognition)
    img = ImageOps.autocontrast(img)

    # Resize to 28×28
    img = img.resize((28, 28))

    # Convert to NumPy
    img = np.array(img)

    # Normalize (0–1)
    img = img / 255.0

    # Reshape for CNN
    img = img.reshape(1, 28, 28, 1)

    # Predict
    pred = model.predict(img)
    prob = float(np.max(pred))
    idx = int(np.argmax(pred))

    char = LABELS[idx]

    # Display results
    st.subheader(f"Prediction: **{char}**")
    st.write(f"Confidence: `{prob*100:.2f}%`")
