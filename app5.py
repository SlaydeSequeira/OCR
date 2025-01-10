import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import io
import base64

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121(pretrained=False)
num_features = model.classifier.in_features
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(num_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(512, 3),
    torch.nn.Softmax(dim=1)
)
model.load_state_dict(torch.load("ai/trash_classifier_best.pth", map_location=device))
model = model.to(device)
model.eval()

# Labels for the classes
labels = {0: "Biodegradable", 1: "Recyclable", 2: "Hazardous"}

# Transformation for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Streamlit App
st.title("Trash Classification")
st.write("Upload an image to classify it as Biodegradable, Recyclable, or Hazardous.")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load the image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess the image for model prediction
        input_image = transform(img).unsqueeze(0).to(device)
        
        # Make predictions
        with torch.no_grad():
            outputs = model(input_image)
            _, predicted_class = torch.max(outputs, 1)
            predicted_label = labels[predicted_class.item()]
        
        # Display the prediction
        st.write(f"### Predicted Trash Type: {predicted_label}")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")