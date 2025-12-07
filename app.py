import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Road Damage Detection", page_icon="🛣️", layout="wide")
st.title("🛣️ Road Damage Detection & Safety Assessment")
st.write("Upload a road image and optional location to receive a full safety analysis.")


# ---------------------------------------------------------
# LOAD MODEL — rebuild architecture + load weights
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
IMG_SIZE = (224, 224)
st.sidebar.success("Model loaded successfully!")


# ---------------------------------------------------------
# DAMAGE INFO (simple English + urgency + danger level)
# ---------------------------------------------------------
CRACK_INFO = {
    0: {
        "name": "Long Crack",
        "desc": "A long, continuous crack usually caused by pavement aging or temperature stress.",
        "danger": "Low",
        "urgency": "Low",
        "repair": "Crack sealing recommended.",
        "action": "Drive cautiously and monitor condition over time."
    },
    1: {
        "name": "Cross Crack",
        "desc": "Cracks crossing the road width. Often expands quickly.",
        "danger": "Medium",
        "urgency": "Medium",
        "repair": "Crack filling and resurfacing may be required.",
        "action": "Report to local authorities for preventive maintenance."
    },
    2: {
        "name": "Spiderweb Crack",
        "desc": "A network of cracks forming a web-like pattern. Early sign of pothole formation.",
        "danger": "High",
        "urgency": "High",
        "repair": "Full-depth patching recommended.",
        "action": "Avoid driving over this area. Authorities have been notified."
    },
    3: {
        "name": "Block Crack",
        "desc": "Cracks forming square or rectangular blocks. Indicates structural road weakness.",
        "danger": "Medium to High",
        "urgency": "High",
        "repair": "Area may require resurfacing or reconstruction.",
        "action": "Reduce speed and report this location if not already documented."
    },
    4: {
        "name": "Other Road Damage",
        "desc": "General road distress detected but unclear category.",
        "danger": "Varies",
        "urgency": "Unknown",
        "repair": "Inspection required.",
        "action": "Use caution and report the road condition."
    }
}

BADGE_COLORS = {
    "Low": "green",
    "Medium": "orange",
    "High": "red",
    "Medium to High": "red",
    "Unknown": "gray",
    "Varies": "gray"
}


# ---------------------------------------------------------
# CACHED PREDICTION
# ---------------------------------------------------------
@st.cache_data
def predict_image(model, img_array):
    preds = model.predict(img_array)
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100
    return class_id, confidence, preds


# ---------------------------------------------------------
# SIDEBAR — LOCATION INPUT
# ---------------------------------------------------------
st.sidebar.header("📍 Enter Location")
address = st.sidebar.text_input("Road address (optional)")

location_coords = None
if address:
    geolocator = Nominatim(user_agent="road_safety_app")
    try:
        loc = geolocator.geocode(address)
        if loc:
            location_coords = (loc.latitude, loc.longitude)
            st.sidebar.success(f"Coordinates: {loc.latitude}, {loc.longitude}")
        else:
            st.sidebar.error("Location not found.")
    except:
        st.sidebar.error("Geolocation failed. Try again.")


# ---------------------------------------------------------
# DISPLAY MAP
# ---------------------------------------------------------
if location_coords:
    st.subheader("📍 Location Map")
    m = folium.Map(location=location_coords, zoom_start=15)
    folium.Marker(location_coords, tooltip="Selected Road").add_to(m)
    st_folium(m, width=700, height=450)


# ---------------------------------------------------------
# IMAGE UPLOAD + PREDICTION
# ---------------------------------------------------------
uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1])

    # ---------------------- IMAGE ----------------------
    with col1:
        st.subheader("📸 Uploaded Image")
        st.image(image, use_column_width=True)

    # ---------------------- PREDICTION -----------------
    with col2:
        st.subheader("🔍 Detection Result")

        # Preprocess
        img = image.resize(IMG_SIZE)
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Prediction
        class_id, confidence, preds = predict_image(model, img)
        info = CRACK_INFO[class_id]

        # Professional Styled Output
        st.markdown(f"### 🛠️ {info['name']} Detected")
        st.write(info["desc"])

        # Danger + Urgency Badges
        st.write(
            f"**Danger Level:** "
            f"<span style='color:{BADGE_COLORS[info['danger']]}; font-weight:bold;'>{info['danger']}</span>",
            unsafe_allow_html=True
        )

        st.write(
            f"**Urgency:** "
            f"<span style='color:{BADGE_COLORS[info['urgency']]}; font-weight:bold;'>{info['urgency']}</span>",
            unsafe_allow_html=True
        )

        st.write(f"**Recommended Repair:** {info['repair']}")
        st.write(f"**Safety Advice:** {info['action']}")

        # Confidence Bar
        st.write("### Model Confidence")
        st.progress(confidence / 100)
        st.write(f"**Confidence Score:** {confidence:.2f}%")

        # Authority Notice
        if info["danger"] in ["High", "Medium to High"]:
            st.warning("🚨 This road damage is severe. If location is provided, local authorities may be alerted.")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.write("© 2025 Road Safety Intelligence System — Final Project Submission (Machine Learning for CV)")
