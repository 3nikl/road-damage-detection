import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Road Damage Detection", page_icon="🛣️", layout="centered")
st.title("🛣️ Road Damage Detection & Classification")
st.write("Upload a road image to detect the crack type and get a repair recommendation.")

IMG_SIZE = (224, 224)

# ---------------------------------------------------------
# LOAD MODEL (NO WARNING VERSION)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        input_shape=(224, 224, 3),
        weights="imagenet"
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    model.load_weights("road_damage_model.keras")
    return model

model = load_model()

# ---------------------------------------------------------
# CLASS LABELS
# ---------------------------------------------------------
CRACK_INFO = {
    0: {
        "name": "Longitudinal Crack",
        "meaning": "A long crack parallel to the road caused by aging or temperature expansion.",
        "repair": "Usually sealed. Not an emergency."
    },
    1: {
        "name": "Transverse Crack",
        "meaning": "Cracks that cut across the road due to thermal stress.",
        "repair": "Should be repaired in 1–2 days to prevent widening."
    },
    2: {
        "name": "Alligator / Spider Crack",
        "meaning": "Interconnected cracks indicating the pavement structure has failed.",
        "repair": "High priority. Area needs full-depth patching."
    },
    3: {
        "name": "Block Crack",
        "meaning": "Cracks forming rectangular blocks due to asphalt shrinkage.",
        "repair": "Moderate priority. Resurfacing recommended."
    },
    4: {
        "name": "Other Distress / Damage",
        "meaning": "Detected road distress not matching the other categories.",
        "repair": "Inspection required to determine repair urgency."
    }
}

# ---------------------------------------------------------
# PREDICTION FUNCTION (NO WARNING)
# ---------------------------------------------------------
@st.cache_data
def predict(img_array):
    preds = model.predict(img_array, verbose=0)
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100
    return class_id, confidence

# ---------------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------------
uploaded = st.file_uploader("📸 Upload a road image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    # NEW — No warning: use_container_width instead of use_column_width
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    class_id, confidence = predict(img)
    info = CRACK_INFO[class_id]

    st.subheader("🔍 Road Damage Analysis")
    st.write(f"**Detected Damage Type:** {info['name']}")
    st.write(f"**What This Means:** {info['meaning']}")
    st.write(f"**Recommended Repair:** {info['repair']}")
    st.write(f"**Model Confidence:** {confidence:.2f}%")

    st.success("This analysis will be forwarded to the appropriate road maintenance authorities.")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.write("---")
st.caption("© 2025 Road Safety Intelligence System — Final CV Project")
