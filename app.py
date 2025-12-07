import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium

# -----------------------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------------------
st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
    layout="centered"
)

# -----------------------------------------
# LOAD MODEL
# -----------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("road_damage_model.keras")
    return model

model = load_model()

# Class names
CLASS_NAMES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Pothole",
    "Surface Damage"
]

# -----------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Upload Image", "Set Location", "Generate Report"]
)

st.sidebar.markdown("---")
st.sidebar.info("Built by Nikhil • Final Project")

# -----------------------------------------
# PAGE 1 — UPLOAD IMAGE & PREDICT
# -----------------------------------------
if page == "Upload Image":

    st.title("🛣️ Road Damage Detection")
    st.write("Upload a road image to detect the type of damage.")

    uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocessing
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, 0)

        # Prediction
        preds = model.predict(img_array)
        class_id = np.argmax(preds)
        confidence = np.max(preds)

        # Results
        st.success(f"**Damage Type:** {CLASS_NAMES[class_id]}")
        st.info(f"Confidence: **{confidence:.2f}**")

        # Save to session
        st.session_state["prediction"] = CLASS_NAMES[class_id]
        st.session_state["confidence"] = float(confidence)

        # Save image to session
        st.session_state["uploaded_image"] = uploaded


# -----------------------------------------
# PAGE 2 — SET LOCATION (ADDRESS OR MAP)
# -----------------------------------------
elif page == "Set Location":

    st.title("📍 Add Road Damage Location")

    st.write("### Option 1: Enter Address")
    address = st.text_input("Street, City, ZIP")

    if st.button("Convert to Coordinates"):
        geolocator = Nominatim(user_agent="road_damage_app")
        location = geolocator.geocode(address)

        if location:
            st.success(f"Coordinates found: {location.latitude}, {location.longitude}")
            st.session_state["lat"] = location.latitude
            st.session_state["lon"] = location.longitude
        else:
            st.error("Address not found. Try again.")

    st.write("---")
    st.write("### Option 2: Drop a Pin on the Map")

    # Default map centered on USA (Boston example)
    m = folium.Map(location=[42.36, -71.06], zoom_start=10)
    map_output = st_folium(m, height=350, width=700)

    if map_output and map_output.get("last_clicked"):
        lat = map_output["last_clicked"]["lat"]
        lon = map_output["last_clicked"]["lng"]

        st.session_state["lat"] = lat
        st.session_state["lon"] = lon

        st.success(f"Selected Location: {lat}, {lon}")


# -----------------------------------------
# PAGE 3 — GENERATE FINAL REPORT
# -----------------------------------------
elif page == "Generate Report":

    st.title("📄 Damage Report Summary")

    if "prediction" not in st.session_state:
        st.warning("No prediction found. Please upload an image first.")
        st.stop()

    # Show uploaded image
    if "uploaded_image" in st.session_state:
        st.image(st.session_state["uploaded_image"], caption="Reported Image", use_column_width=True)

    # Show prediction
    st.write("### 🔍 Damage Type Detected:")
    st.success(st.session_state["prediction"])

    st.write("### 📊 Model Confidence:")
    st.info(f"{st.session_state['confidence']:.2f}")

    # Location info
    st.write("### 📍 Location Details:")

    if "lat" in st.session_state and "lon" in st.session_state:
        st.write(f"Latitude: **{st.session_state['lat']}**")
        st.write(f"Longitude: **{st.session_state['lon']}**")
    else:
        st.warning("No location added yet.")

    # Recommended actions
    st.write("### 🛠️ Recommended Action:")
    st.write("""
- **Pothole** → Immediate maintenance required  
- **Alligator Crack** → Structural repair recommended  
- **Longitudinal / Transverse Cracks** → Seal to prevent spreading  
- **Surface Damage** → Monitor regularly; low priority  
""")

    # Download report
    report_text = f"""
ROAD DAMAGE REPORT

Damage Type: {st.session_state['prediction']}
Confidence: {st.session_state['confidence']:.2f}

Location:
Latitude: {st.session_state.get('lat', 'N/A')}
Longitude: {st.session_state.get('lon', 'N/A')}
"""

    st.download_button(
        "📥 Download Report",
        data=report_text,
        file_name="road_damage_report.txt"
    )
