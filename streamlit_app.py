import cv2
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
# For inference, we use deterministic transforms (no random augmentations)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

st.set_page_config(page_title="Driver Monitoring System", page_icon="🚗", layout="wide")

# Sidebar
st.sidebar.title("ℹ️ About the App")
st.sidebar.write("This system detects driver distractions using **YOLOv11 + CNN + SVM**.")
st.sidebar.write("⚠️ **Alerts are triggered** if unsafe behaviors are detected.")

# Title and Description
st.markdown("<h1 style='text-align: center; color: #FF5733;'>🚗 Driver Behavior Monitoring</h1>", unsafe_allow_html=True)
st.write("🔍 **Real-time driver distraction detection using AI models.**")

# Buttons in a row
col1, col2 = st.columns(2)
start_webcam = col1.button("▶️ Start Webcam", key="start", help="Start the driver monitoring system")
stop_webcam = col2.button("⏹️ Stop Webcam", key="stop", help="Stop the webcam feed")

# Webcam Display
stframe = st.empty()
status_placeholder = st.empty()  # Status Message

# ---------------- Main Processing ----------------
if start_webcam:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            status_placeholder.error("🚫 Error: Cannot access webcam")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            person_crop = image_rgb[y1:y2, x1:x2]

            if person_crop.shape[0] > 0 and person_crop.shape[1] > 0:
                image_tensor = transform(person_crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    features = feature_extractor(image_tensor)
                features = features.view(features.size(0), -1).cpu().numpy()

                # Predict with SVM (without probabilities)
                prediction = svm_model.predict(features)[0]
                detected_label = class_labels[prediction]

                # Set color based on behavior (green for normal, red otherwise)
                color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{detected_label}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                if prediction != 0:
                    alert_triggered = True
                    play_alert()
                    time.sleep(3)
        else:
            detected_label = "No person detected"

        # Display status in Streamlit
        if alert_triggered:
            status_placeholder.error(f"🚨 **ALERT:** {detected_label} detected!")
        else:
            status_placeholder.success("✅ **Status: Normal Driving**")

        # Display webcam feed
        stframe.image(frame, channels="BGR", use_column_width=True)

        if stop_webcam:
            break

    cap.release()
    st.warning("🔴 Webcam Stopped.")

#--------Upload Photo-------------------
    st.subheader("Class Labels for Driver Behavior Classification:")
for key, value in class_labels.items():
    st.write(f"**{key}: {value}**")

# Upload Image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_rgb = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # --------------- YOLOv11: Person Detection ---------------
    results = yolo_model(image_rgb)

    # List to store all person detections
    person_boxes = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls.item())  # Get class ID
            if cls == 0:  # Class ID 0 corresponds to "Person"
                # Save the box and its confidence score
                person_boxes.append(box)

    if person_boxes:
        # Select the box with the highest confidence
        best_box = max(person_boxes, key=lambda b: b.conf.item())
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())

        # Crop detected person
        person_crop = image_rgb[y1:y2, x1:x2]

        # Ensure it's not empty
        if person_crop.shape[0] > 0 and person_crop.shape[1] > 0:
            # Preprocess Image for Behavior Model
            image_tensor = transform(person_crop).unsqueeze(0).to(device)

            # Extract Features using CNN
            with torch.no_grad():
                features = feature_extractor(image_tensor)
            features = features.view(features.size(0), -1).cpu().numpy()

            # Predict with SVM
            prediction = svm_model.predict(features)[0]
            predicted_label = class_labels[prediction]

            # Display Result
            st.write(f"### 🚦 Predicted Activity: {predicted_label}")
            # if prediction != 0:  # 0 corresponds to "Normal Driving"
            #     play_alert()
    else:
        st.write("No person detected.")
