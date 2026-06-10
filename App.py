import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image

CLASS_NAMES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor (Healthy)", "Pituitary Tumor"]

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 4)
    )

    model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

st.set_page_config(page_title="Brain MRI Classifier", page_icon="🧠")

st.title("🧠 Brain MRI Tumor Classifier")
st.write("Upload a 2D Brain MRI scan, and the model will predict the tumor type.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
    
    st.write("Analyzing...")

    input_tensor = transform(image).unsqueeze(0)
    
    with torch.inference_mode():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item() * 100
    
    st.subheader("Results:")
    prediction = CLASS_NAMES[predicted_idx]
    
    if "No Tumor" in prediction:
        st.success(f"Prediction: **{prediction}**")
    else:
        st.error(f"Prediction: **{prediction}**")
        
    st.info(f"Confidence: **{confidence:.2f}%**")