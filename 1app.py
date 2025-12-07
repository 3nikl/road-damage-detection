import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, 3)  # 3 classes
    model.load_state_dict(torch.load("road_damage_pytorch.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -------------------------------
# Preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -------------------------------
# Labels / Severity
# -------------------------------
INFO = {
    0: {
        "name": "Long Crack",
        "danger": "Medium",
        "urgency": "Fix within 3–5 days",
        "advice": "Condition needs attention soon but not critical."
    },
    1: {
        "name": "Cross Crack",
        "danger": "High",
        "urgency": "Fix within 24 hours",
        "advice": "Crack progresses fast. Authorities should be notified."
    },
    2: {
        "name": "Severe Damage / Pothole",
        "danger": "Critical",
        "urgency": "Urgent — Repair Now",
        "advice": "High risk for vehicles. Emergency maintenance required."
    }
}

COLOR = {
    "Medium": "🟡",
    "High": "🟠",
    "Critical": "🔴"
}

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Road Damage Detection", page_icon="🚧")
st.title("🚧 Road Damage Detection System")
st.write("Upload a road image to detect cracks and classify the risk level.")

# -------------------------------
# Location Input
# -------------------------------
st.subheader("📍 Optional: Enter Location")
col_lat, col_long = st.columns(2)

latitude = col_lat.text_input("Latitude")
longitude = col_long.text_input("Longitude")

if latitude and longitude:
    st.success(f"Location received: ({latitude}, {longitude})")

# -------------------------------
# Image Upload
# -------------------------------
st.subheader("📸 Upload Road Image")
uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    x = transform(img).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        preds = model(x)
        probs = torch.softmax(preds, dim=1)
        class_id = int(torch.argmax(probs))
        confidence = float(probs[0][class_id]) * 100

    details = INFO[class_id]

    # -------------------------------
    # Output Card (NO HTML)
    # -------------------------------
    st.subheader("🔍 Detection Result")
    st.write(f"**Detected:** {details['name']}")
    st.write(f"**Danger Level:** {COLOR[details['danger']]} {details['danger']}")
    st.write(f"**Urgency:** {details['urgency']}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.write(f"**Recommendation:** {details['advice']}")

    # Location + severity combined message
    if latitude and longitude:
        st.info(
            f"📍 This damage location ({latitude}, {longitude}) has been flagged as **{details['danger']} severity**.\n\n"
            f"It is recommended to notify local Boston road authorities."
        )

st.write("---")
st.caption("Built with Streamlit + PyTorch — Road Safety Project")

