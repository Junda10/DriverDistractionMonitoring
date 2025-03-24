import torch
import torch.nn as nn
import torchvision.transforms as transforms
import joblib
import pygame
import streamlit as st
import time
from ultralytics import YOLO
from PIL import Image
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# ---------------- Alert Sound ----------------
pygame.mixer.init()
pygame.mixer.music.load("preview.mp3")

def play_alert():
    pygame.mixer.music.play()

# ---------------- Load Trained Models ----------------

# Define CNN Model for Behavior Classification
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ImprovedCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the features
        x = self.fc_layers(x)
        return x

# Class Labels Mapping for Behavior Classification
class_labels = {
    0: "Normal Driving",
    1: "Texting - Right Hand",
    2: "Talking on Phone - Right Hand",
    3: "Texting - Left Hand",
    4: "Talking on Phone - Left Hand",
    5: "Operating the Radio",
    6: "Drinking",
    7: "Reaching Behind",
    8: "Hair and Makeup",
    9: "Talking to Passenger"
}

# Load YOLOv11 Model for Person Detection
yolo_model = YOLO("yolo11n.pt")  # Ensure this file exists

# Load Pretrained CNN Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = ImprovedCNN(num_classes=10).to(device)
cnn_model.load_state_dict(torch.load("best_model_CNN_95.53.pth", map_location=device))
cnn_model.eval()

# Create Feature Extractor: Use conv_layers and explicitly flatten the output
feature_extractor = nn.Sequential(
    cnn_model.conv_layers,
    nn.Flatten()
).to(device)

# Load Trained SVM Model (without probability enabled)
svm_model = joblib.load("svm_classifier_gridsearch.pkl")

# ---------------- Image Preprocessing ----------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------- Streamlit UI Setup ----------------
st.set_page_config(page_title="Driver Monitoring System", page_icon="🚗", layout="wide")

# Sidebar Information
st.sidebar.title("ℹ️ About the App")
st.sidebar.write("This system detects driver distractions using **YOLOv11 + CNN + SVM**.")
st.sidebar.write("⚠️ **Alerts are triggered** if unsafe behaviors are detected.")

# Title and Description
st.markdown("<h1 style='text-align: center; color: #FF5733;'>🚗 Driver Behavior Monitoring</h1>", unsafe_allow_html=True)
st.write("🔍 **Real-time driver distraction detection using AI models.**")

# ---------------- Video Processing via streamlit-webrtc ----------------
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        # Convert frame to a NumPy array in BGR format
        img = frame.to_ndarray(format="bgr24")
        # Convert BGR to RGB for processing
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = yolo_model(image_rgb)

        detected_label = "Normal Driving"
        alert_triggered = False

        # Process YOLO detections: collect all person boxes
        person_boxes = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls.item())
                if cls == 0:  # Only process person detections
                    person_boxes.append(box)

        if person_boxes:
            # Select the box with the highest confidence
            best_box = max(person_boxes, key=lambda b: b.conf.item())
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
            # Draw rectangle on the frame
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Crop the detected person
            person_crop = image_rgb[y1:y2, x1:x2]
            if person_crop.shape[0] > 0 and person_crop.shape[1] > 0:
                image_tensor = transform(person_crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    features = feature_extractor(image_tensor)
                features = features.view(features.size(0), -1).cpu().numpy()
                # Predict behavior using the SVM model
                prediction = svm_model.predict(features)[0]
                detected_label = class_labels[prediction]

                # Set color based on behavior (green for normal, red otherwise)
                color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{detected_label}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                if prediction != 0:
                    alert_triggered = True
                    play_alert()
                    # Delay slightly to prevent rapid-fire alerts
                    time.sleep(0.5)
        else:
            detected_label = "No person detected"

        # Display status text on the frame
        cv2.putText(img, f"Status: {detected_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(key="driver-monitoring", video_transformer_factory=VideoProcessor, rtc_configuration=rtc_configuration)

# ---------------- Image Upload Processing ----------------
st.subheader("Upload an Image for Driver Behavior Analysis")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_rgb = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # YOLOv11 Person Detection on Uploaded Image
    results = yolo_model(image_rgb)
    person_boxes = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls.item())
            if cls == 0:  # Class 0 corresponds to "Person"
                person_boxes.append(box)

    if person_boxes:
        best_box = max(person_boxes, key=lambda b: b.conf.item())
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        person_crop = image_rgb[y1:y2, x1:x2]
        if person_crop.shape[0] > 0 and person_crop.shape[1] > 0:
            image_tensor = transform(person_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                features = feature_extractor(image_tensor)
            features = features.view(features.size(0), -1).cpu().numpy()
            prediction = svm_model.predict(features)[0]
            predicted_label = class_labels[prediction]
            st.write(f"### 🚦 Predicted Activity: {predicted_label}")
    else:
        st.write("No person detected.")

# ---------------- Display Class Labels in Sidebar ----------------
st.sidebar.subheader("Class Labels for Driver Behavior Classification:")
for key, value in class_labels.items():
    st.sidebar.write(f"**{key}: {value}**")
