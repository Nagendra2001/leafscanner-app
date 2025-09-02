import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image

# Title and page setup
st.set_page_config(page_title="LeafScanner 🌿", layout="centered")
st.title("🌿 Plant Disease Prediction App")

# Class labels
class_names = [
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Chatbot advice
disease_solutions = {
    "Tomato_Bacterial_spot": "Use copper-based fungicides and avoid overhead watering.",
    "Tomato_Early_blight": "Remove infected leaves and apply fungicide with chlorothalonil.",
    "Tomato_Late_blight": "Use certified seeds, remove infected plants, apply mancozeb fungicide.",
    "Tomato_Leaf_Mold": "Improve air circulation and apply sulfur-based fungicide.",
    "Tomato_Septoria_leaf_spot": "Prune affected leaves and apply neem oil or copper spray.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Use miticides or neem oil. Increase humidity.",
    "Tomato__Target_Spot": "Remove debris and apply fungicides containing azoxystrobin.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Control whiteflies using yellow sticky traps.",
    "Tomato__Tomato_mosaic_virus": "Remove infected plants. Disinfect tools regularly.",
    "Tomato_healthy": "No disease detected! Keep up the good care."
}

# Fertilizer NPK info
fertilizer_info = {
    "Tomato_Bacterial_spot": "Nitrogen: 20%, Phosphorus: 10%, Potassium: 20%",
    "Tomato_Early_blight": "Nitrogen: 25%, Phosphorus: 15%, Potassium: 30%",
    "Tomato_Late_blight": "Nitrogen: 15%, Phosphorus: 20%, Potassium: 25%",
    "Tomato_Leaf_Mold": "Nitrogen: 20%, Phosphorus: 20%, Potassium: 20%",
    "Tomato_Septoria_leaf_spot": "Nitrogen: 18%, Phosphorus: 12%, Potassium: 25%",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Nitrogen: 20%, Phosphorus: 15%, Potassium: 30%",
    "Tomato__Target_Spot": "Nitrogen: 22%, Phosphorus: 18%, Potassium: 28%",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Nitrogen: 25%, Phosphorus: 20%, Potassium: 30%",
    "Tomato__Tomato_mosaic_virus": "Nitrogen: 23%, Phosphorus: 17%, Potassium: 25%",
    "Tomato_healthy": "Fertilizer: Balanced NPK 10-10-10"
}

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(class_names))
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
model.to(device)

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# File uploader
uploaded_file = st.file_uploader("📷 Upload a tomato leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
        predicted_class = class_names[pred.item()]
        confidence = torch.nn.functional.softmax(output, dim=1)[0][pred.item()] * 100

    with col2:
        st.markdown(f"### 🩺 Predicted Disease:\n**{predicted_class}**")
        st.markdown(f"### 🔬 Confidence:\n**{confidence:.2f}%**")

    # Expander for Chatbot Advice
    with st.expander("🤖 Chatbot Advice"):
        st.info(disease_solutions.get(predicted_class, "No advice available."))

    # Expander for Fertilizer Info
    with st.expander("🌾 Fertilizer Recommendation (NPK Ratio)"):
        st.success(fertilizer_info.get(predicted_class, "No fertilizer data available."))

    st.success("✅ Prediction Complete. Scroll down for details.")
