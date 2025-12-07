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
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 3)   # 3 classes
    model.load_state_dict(torch.load("road_damage_pytorch.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -------------------------------
# Image Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -------------------------------
# Class Labels & Metadata
# -------------------------------
LABELS = {
    0: {
        "name": "Long Crack",
        "danger": "Medium",
        "urgency": "Fix within 3–5 days",
        "message": "Report submitted to Boston City Maintenance."
    },
    1: {
        "name": "Cross Crack",
        "danger": "High",
        "urgency": "Fix within 24 hours",
        "message": "Immediate alert sent to City Repair Services."
    },
    2: {
        "name": "Severe Damage / Pothole",
        "danger": "Critical",
        "urgency": "URGENT: Repair Now",
        "message": "Emergency ticket created. Maintenance crew notified."
    }
}

DANGER_COLOR = {
    "Medium": "orange",
    "High": "red",
    "Critical": "darkred"
}

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Road Damage Detection", page_icon="🚧", layout="centered")
st.title("🚧 Boston Smart Road Damage Detection")
st.write("Upload a road image to detect cracks and classify risk level.")

uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    x = transform(img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        pred = model(x)
        probs = torch.softmax(pred, dim=1)
        class_id = int(torch.argmax(probs))
        confidence = float(probs[0][class_id]) * 100

    info = LABELS[class_id]

    # -------------------------------
    # Display Result Card
    # -------------------------------
    st.markdown(f"""
    <div style="
        padding:20px;
        border-radius:12px;
        border:1px solid #ddd;
        background-color:#fafafa;">
        
        <h2 style="margin-bottom:5px;">{info['name']}</h2>

        <p><b>Danger Level:</b> 
            <span style="color:{DANGER_COLOR[info['danger']]}; font-weight:bold;">
                {info['danger']}
            </span>
        </p>

        <p><b>Urgency:</b> {info['urgency']}</p>
        
        <p><b>System Message:</b> {info['message']}</p>

        <p><b>Confidence:</b> {confidence:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

st.write("---")
st.write("Built with ❤️ using PyTorch + Streamlit")
