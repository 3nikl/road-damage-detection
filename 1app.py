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
    model.classifier[1] = nn.Linear(model.last_channel, 3)
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
# Crack Explanations (No HTML, No emojis)
# -------------------------------
EXPLANATION = {
    0: {
        "name": "Longitudinal Crack",
        "meaning": "A long crack running parallel to the road. It usually appears because of road aging, weak joints, or continuous vehicle load.",
        "cause": "Temperature changes and shrinkage of asphalt cause long, straight cracks that slowly expand.",
        "repair": "Not extremely urgent but should be sealed to prevent water from entering and creating potholes later."
    },
    1: {
        "name": "Transverse / Cross Crack",
        "meaning": "A crack that runs across the width of the road. These cracks are early indicators of structural weakness.",
        "cause": "Expansion and contraction due to temperature changes make the asphalt break across the road.",
        "repair": "Repair is recommended soon because cross cracks grow faster and can turn into potholes."
    },
    2: {
        "name": "Severe Road Damage / Pothole Formation",
        "meaning": "This indicates high road distress. The crack pattern suggests the asphalt surface has weakened deeply.",
        "cause": "Moisture, repeated traffic stress, and delayed maintenance break the bond between asphalt layers.",
        "repair": "Immediate attention is required because this type of damage can harm vehicles and expand rapidly."
    }
}

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Road Damage Detection", page_icon="🚧")
st.title("Road Damage Identification System")
st.write("Upload a road image to automatically identify the type of crack and receive a meaningful explanation.")

# -------------------------------
# OPTIONAL LOCATION INPUT
# -------------------------------
st.subheader("Optional: Enter Location")
col1, col2 = st.columns(2)
latitude = col1.text_input("Latitude")
longitude = col2.text_input("Longitude")

if latitude and longitude:
    st.write(f"Location recorded: ({latitude}, {longitude})")

# -------------------------------
# IMAGE UPLOAD
# -------------------------------
st.subheader("Upload Road Image")
uploaded = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    x = transform(img).unsqueeze(0)

    # Model prediction
    with torch.no_grad():
        preds = model(x)
        probs = torch.softmax(preds, dim=1)
        class_id = int(torch.argmax(probs))
        confidence = float(probs[0][class_id]) * 100

    info = EXPLANATION[class_id]

    # -------------------------------
    # RESULTS (Clean text only)
    # -------------------------------
    st.subheader("Detection Result")
    st.write(f"Identified Type: **{info['name']}**")
    st.write(f"Explanation: {info['meaning']}")
    st.write(f"Why This Happens: {info['cause']}")
    st.write(f"Recommended Action: {info['repair']}")
    st.write(f"Model Confidence: {confidence:.2f}%")

    if latitude and longitude:
        st.write(
            f"This road damage at location ({latitude}, {longitude}) "
            f"has been documented and analyzed."
        )

    st.write(
        "Thank you. This assessment will be forwarded to the appropriate road authorities."
    )

st.write("---")
st.caption("Road Damage Analysis System • Built with PyTorch + Streamlit")
