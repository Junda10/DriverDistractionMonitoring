import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import joblib
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

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
        x = x.view(x.size(0), -1)  # Flatten
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

# Load YOLO Model
yolo_model = YOLO("yolo11n.pt")  # Make sure this file exists in your repo

# Load Pretrained CNN Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = ImprovedCNN(num_classes=10).to(device)
cnn_model.load_state_dict(torch.load("best_model_CNN_95.53.pth", map_location=device))
cnn_model.eval()

# Create Feature Extractor: Use conv_layers and flatten output
feature_extractor = nn.Sequential(
    cnn_model.conv_layers,
    nn.Flatten()
).to(device)

# Load Trained SVM Model
svm_model = joblib.load("svm_classifier_gridsearch_95.53.pkl")

# ---------------- Image Preprocessing ----------------
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

st.markdown("<h1 style='text-align: center; color: #FF5733;'>🚗 Driver Behavior Monitoring</h1>", unsafe_allow_html=True)
st.write("🔍 **Real-time driver distraction detection using AI models.**")

# ---------------- Real-time Video with streamlit-webrtc ----------------# ---------------- Real-time Video with streamlit-webrtc ----------------
# Debug log function
def debug_log(message):
    print(message)
    # Optionally, write to a file or use st.write() in a placeholder if you want to see them in the app.

# ---------------- Real-time Video Processing via streamlit-webrtc ----------------
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        # Convert frame to NumPy array in BGR format
        img = frame.to_ndarray(format="bgr24")
        debug_log("Frame received for processing. Shape: {}".format(img.shape))
        
        # Convert BGR to RGB
        try:
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            debug_log("Error in cvtColor: {}".format(e))
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Run YOLO detection
        try:
            results = yolo_model(image_rgb)
            debug_log("YOLO results: {}".format(results))
        except Exception as e:
            debug_log("Error running YOLO: {}".format(e))
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        detected_label = "Normal Driving"
        person_boxes = []
        
        # Iterate over YOLO results and collect boxes for class 0 (person)
        for result in results:
            debug_log("Processing a result...")
            for box in result.boxes:
                try:
                    cls = int(box.cls.item())
                    conf = box.conf.item()
                    debug_log("Detected class: {} with confidence: {}".format(cls, conf))
                    if cls == 0:  # Check if this is a person
                        person_boxes.append(box)
                except Exception as e:
                    debug_log("Error processing a box: {}".format(e))
        
        debug_log("Number of person boxes detected: {}".format(len(person_boxes)))
        
        if person_boxes:
            # Select the box with highest confidence
            best_box = max(person_boxes, key=lambda b: b.conf.item())
            try:
                x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
                debug_log("Best box coordinates: {} {} {} {}".format(x1, y1, x2, y2))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            except Exception as e:
                debug_log("Error getting box coordinates: {}".format(e))
            
            # Crop detected region
            person_crop = image_rgb[y1:y2, x1:x2]
            debug_log("Cropped person region shape: {}".format(person_crop.shape))
            if person_crop.shape[0] > 0 and person_crop.shape[1] > 0:
                try:
                    tensor = transform(person_crop).unsqueeze(0).to(device)
                except Exception as e:
                    debug_log("Error transforming person crop: {}".format(e))
                    tensor = None
                if tensor is not None:
                    with torch.no_grad():
                        features = feature_extractor(tensor)
                    features = features.view(features.size(0), -1).cpu().numpy()
                    try:
                        prediction = svm_model.predict(features)[0]
                        detected_label = class_labels[prediction]
                        debug_log("Predicted behavior: {}".format(detected_label))
                    except Exception as e:
                        debug_log("Error in SVM prediction: {}".format(e))
                        detected_label = "Error predicting"
                    
                    color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, detected_label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            detected_label = "No person detected"
            debug_log("No person detected in this frame.")
        
        cv2.putText(img, f"Status: {detected_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_streamer(key="driver-monitoring", video_transformer_factory=VideoProcessor,
                rtc_configuration=rtc_configuration)

# ---------------- Image Upload Processing ----------------
st.subheader("Upload an Image for Driver Behavior Analysis")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
import base64
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
            if prediction != 0:  # 0 corresponds to "Normal Driving"
                with open("preview.mp3", "rb") as f:
                    audio_bytes = f.read()
                audio_b64 = base64.b64encode(audio_bytes).decode()

                # Create an HTML audio element with autoplay enabled
                audio_html = f"""
                <audio autoplay>
                    <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                </audio>
                """

                # Render the HTML
                st.markdown(audio_html, unsafe_allow_html=True)
    else:
        st.write("No person detected.")

# ---------------- Class Labels in Sidebar ----------------
st.sidebar.subheader("Class Labels for Driver Behavior Classification:")
for key, value in class_labels.items():
    st.sidebar.write(f"**{key}: {value}**")
