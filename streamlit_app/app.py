import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import streamlit as st
import gdown

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model file path and download URL
model_path = "../saved_models/trained_model.pth"
model_url = "https://drive.google.com/uc?id=1PLQ6k1ieFMvkrTy8raNvxTWXADZwhH5H"  # Update with the actual Google Drive link

# Download model if not available
if not os.path.exists(model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    gdown.download(model_url, model_path, quiet=False)

# Define the model architecture
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256 * 16 * 16, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 38)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the trained model
model = ImprovedCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

# Class names
class_names = [
    'Apple - Apple scab',
    'Apple - Black rot',
    'Apple - Cedar apple rust',
    'Apple - Healthy',
    'Blueberry - Healthy',
    'Cherry (including sour) - Powdery mildew',
    'Cherry (including sour) - Healthy',
    'Corn (maize) - Cercospora leaf spot / Gray leaf spot',
    'Corn (maize) - Common rust',
    'Corn (maize) - Northern Leaf Blight',
    'Corn (maize) - Healthy',
    'Grape - Black rot',
    'Grape - Esca (Black Measles)',
    'Grape - Leaf blight (Isariopsis Leaf Spot)',
    'Grape - Healthy',
    'Orange - Huanglongbing (Citrus greening)',
    'Peach - Bacterial spot',
    'Peach - Healthy',
    'Pepper, bell - Bacterial spot',
    'Pepper, bell - Healthy',
    'Potato - Early blight',
    'Potato - Late blight',
    'Potato - Healthy',
    'Raspberry - Healthy',
    'Soybean - Healthy',
    'Squash - Powdery mildew',
    'Strawberry - Leaf scorch',
    'Strawberry - Healthy',
    'Tomato - Bacterial spot',
    'Tomato - Early blight',
    'Tomato - Late blight',
    'Tomato - Leaf Mold',
    'Tomato - Septoria leaf spot',
    'Tomato - Spider mites / Two-spotted spider mite',
    'Tomato - Target Spot',
    'Tomato - Tomato Yellow Leaf Curl Virus',
    'Tomato - Tomato mosaic virus',
    'Tomato - Healthy'
]

class_map = {i: class_name for i, class_name in enumerate(class_names)}

# Define image preprocessing function
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        normalize
    ])
    return transform(image).unsqueeze(0)

# Define prediction function
def predict(image):
    image_tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return class_map[predicted.item()]

# Streamlit UI
icon_path = os.path.join(os.path.dirname(__file__), 'images', 'im1.png')
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon=icon_path, 
    layout="wide"
)
# Create three columns, with the middle column being the largest
col1, col2, col3 = st.columns([1, 4, 1])  # Adjust the width ratios as needed

# Left Column - Design or Image Placeholder
with col1:
    # Add your design or image here
    st.image(os.path.join(os.path.dirname(__file__), 'images', 'border.jpg'), use_column_width=True)  # Adjust the image source and options
# Center Column - Main Content
with col2:
    st.markdown(
        """
        <h1 style='text-align: center; font-size: 3rem; font-family: "Roboto", sans-serif;'>🪴Diagnose Your Plants🪴</h1>
        <h2 style='text-align: center; font-size: 1.7rem; font-family: "Open Sans", sans-serif;'>Instant disease detection through image analysis</h2>
        """,
        unsafe_allow_html=True
    )
    st.image(os.path.join(os.path.dirname(__file__), 'images', 'banner.png'), use_column_width=True)
    uploaded_file = st.file_uploader("Upload an image of the plant leaf", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    # With type=["png", "jpg", "jpeg"] in st.file_uploader, users will only see options for those supported file types in the upload dialog,

    # Button styling
    st.markdown(
        """
        <style>
        div.stButton > button {
            background-color: #00FF9C;
            color: #31511E;
            border: none;
            border-radius: 40px;
            padding: 20px 45px;
            text-align: center;
            margin: 10px;
            cursor: pointer;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        div.stButton > button:hover {
            background-color: white;
            color: black;
            border: 2px solid #4CAF50;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col_center1, col_center2 = st.columns([3, 3])  # Columns within the center column for image and results

        with col_center1:
            st.image(image, caption="Uploaded Image", width=310)

        with col_center2:
            if st.button("Detect", key="detect_button"):
                # Make prediction (assuming 'predict' is a predefined function)
                prediction = predict(image)
                st.write(f"Detected Disease  ➡️  {prediction}")

# Right Column - Design or Image Placeholder
with col3:
    # Add your design or image here
    st.image(os.path.join(os.path.dirname(__file__), 'images', 'border.png'), use_column_width=True)  # Adjust the image source and options



