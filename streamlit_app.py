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
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import base64
import os
import tempfile
from collections import Counter
# ---------------- Model Definitions & Loading ----------------

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

# Load YOLO Model (ensure the file exists)
yolo_model = YOLO("yolo11n.pt")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Pretrained CNN Model
cnn_model = ImprovedCNN(num_classes=10).to(device)
cnn_model.load_state_dict(torch.load("best_model_CNN_95.53.pth", map_location=device))
cnn_model.eval()

# Create Feature Extractor from CNN (using conv_layers)
feature_extractor = nn.Sequential(
    cnn_model.conv_layers,
    nn.Flatten()
).to(device)

# Load Trained SVM Model
svm_model = joblib.load("svm_classifier_gridsearch.pkl")

# ---------------- Image Preprocessing ----------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------- Video Functions ----------------
def _overlay_text(img, texts):
    """Overlay multiple lines of text on the image."""
    y0, dy = 50, 20
    for i, text in enumerate(texts):
        y = y0 + i * dy
        cv2.putText(img, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

def process_frame(img):
    """
    Process a single frame (numpy array) and return a tuple:
    (annotated frame, predicted label)
    """
    debug_text = [f"Frame shape: {img.shape}"]
    try:
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        debug_text.append(f"cvtColor error: {e}")
        _overlay_text(img, debug_text)
        return img, "Error"

    try:
        results = yolo_model(image_rgb)
    except Exception as e:
        debug_text.append(f"YOLO error: {e}")
        _overlay_text(img, debug_text)
        return img, "Error"

    detected_label = "Normal Driving"
    person_boxes = []
    # Process YOLO results for persons (assumed class 0)
    for result in results:
        for box in result.boxes:
            try:
                cls = int(box.cls.item())
                if cls == 0:
                    person_boxes.append(box)
            except Exception as e:
                debug_text.append(f"Box processing error: {e}")
    debug_text.append(f"Person boxes: {len(person_boxes)}")

    if person_boxes:
        best_box = max(person_boxes, key=lambda b: b.conf.item())
        try:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except Exception as e:
            debug_text.append(f"Coordinate error: {e}")
            x1 = y1 = x2 = y2 = 0
        person_crop = image_rgb[y1:y2, x1:x2]
        if person_crop.shape[0] > 0 and person_crop.shape[1] > 0:
            try:
                tensor = transform(person_crop).unsqueeze(0).to(device)
            except Exception as e:
                tensor = None
            if tensor is not None:
                try:
                    with torch.no_grad():
                        features = feature_extractor(tensor)
                except Exception as e:
                    debug_text.append(f"Feature extraction error: {e}")
                    features = None
                if features is not None:
                    features = features.view(features.size(0), -1).cpu().numpy()
                    try:
                        prediction = svm_model.predict(features)[0]
                        detected_label = class_labels[prediction]
                    except Exception as e:
                        debug_text.append(f"SVM error: {e}")
                        detected_label = "Error predicting"
                    color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, detected_label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        detected_label = "No person detected"
        debug_text.append("No person detected")

    cv2.putText(img, f"Status: {detected_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    _overlay_text(img, debug_text)
    return img, detected_label

def summarize_detection_log(detection_log):
    """Summarize contiguous frame detections with the same activity.
    
    Args:
        detection_log (list of dict): Each dict has keys "Frame" and "Activity".
        
    Returns:
        list of dict: Each dict has keys "Frame Range" and "Activity".
    """
    if not detection_log:
        return []
    summarized = []
    current_activity = detection_log[0]["Activity"]
    start_frame = detection_log[0]["Frame"]
    end_frame = start_frame
    for entry in detection_log[1:]:
        if entry["Activity"] == current_activity:
            end_frame = entry["Frame"]
        else:
            summarized.append({"Frame Range": f"{start_frame} - {end_frame}", "Activity": current_activity})
            current_activity = entry["Activity"]
            start_frame = entry["Frame"]
            end_frame = start_frame
    summarized.append({"Frame Range": f"{start_frame} - {end_frame}", "Activity": current_activity})
    return summarized

# ---------------- Custom Video Processor for Live Tracking ----------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        debug_text = [f"Frame: {self.frame_count}, Shape: {img.shape}"]

        # Convert BGR to RGB for YOLO
        try:
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            debug_text.append(f"cvtColor error: {e}")
            self._overlay_text(img, debug_text)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Run YOLO detection
        try:
            results = yolo_model(image_rgb)
        except Exception as e:
            debug_text.append(f"YOLO error: {e}")
            self._overlay_text(img, debug_text)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        detected_label = "Normal Driving"
        person_boxes = []
        # Process YOLO results for persons (class 0)
        for result in results:
            for box in result.boxes:
                try:
                    cls = int(box.cls.item())
                    conf = box.conf.item()
                    if cls == 0:  # Person class
                        person_boxes.append(box)
                except Exception as e:
                    debug_text.append(f"Box processing error: {e}")
        debug_text.append(f"Person boxes: {len(person_boxes)}")

        if person_boxes:
            best_box = max(person_boxes, key=lambda b: b.conf.item())
            try:
                x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            except Exception as e:
                debug_text.append(f"Coordinate error: {e}")
                x1 = y1 = x2 = y2 = 0
            person_crop = image_rgb[y1:y2, x1:x2]
            
            if person_crop.shape[0] > 0 and person_crop.shape[1] > 0:
                try:
                    tensor = transform(person_crop).unsqueeze(0).to(device)
                except Exception as e:
                    tensor = None
                if tensor is not None:
                    try:
                        with torch.no_grad():
                            features = feature_extractor(tensor)
                    except Exception as e:
                        debug_text.append(f"Feature extraction error: {e}")
                        features = None
                    if features is not None:
                        features = features.view(features.size(0), -1).cpu().numpy()
                        try:
                            prediction = svm_model.predict(features)[0]
                            detected_label = class_labels[prediction]
                        except Exception as e:
                            debug_text.append(f"SVM error: {e}")
                            detected_label = "Error predicting"
                        color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, detected_label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            detected_label = "No person detected"
            debug_text.append("No person detected")

        cv2.putText(img, f"Status: {detected_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self._overlay_text(img, debug_text)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _overlay_text(self, img, texts):
        y0, dy = 50, 20
        for i, text in enumerate(texts):
            y = y0 + i * dy
            cv2.putText(img, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

# ---------------- Streamlit App Layout & Navigation ----------------

st.set_page_config(page_title="Driver Monitoring System", page_icon="🚗", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Page", ["Main Page", "Detection Page"])

if page == "Main Page":
    st.header("Project Overview")
    st.write("""
        **Driver Behavior Monitoring System**  
        This project uses YOLOv11 for person detection combined with a custom CNN-SVM pipeline to identify driver distractions in real time.  
        The system alerts the driver when unsafe behavior is detected.
    """)
    st.write("Use the **Detection Page** from the sidebar to access live tracking or image-based detection.")

elif page == "Detection Page":
    st.header("Driver Distraction Detection")
    detection_mode = st.selectbox("Select Detection Mode", ["Live Tracking", "Photos", "Video Detection"])

    if detection_mode == "Live Tracking":
        st.write("### Live Tracking")
        st.write("This mode uses your webcam for real-time driver behavior monitoring.")
        # Modified webrtc_streamer call with media stream constraints.
        webrtc_streamer(
            key="driver-monitoring",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            frontend_rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

    elif detection_mode == "Photos":
        st.write("### Photo Detection")
        st.subheader("Upload an Image for Driver Behavior Analysis")
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_rgb = np.array(image)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # YOLO detection on the uploaded image
            results = yolo_model(image_rgb)
            person_boxes = []
            for result in results:
                for box in result.boxes:
                    try:
                        cls = int(box.cls.item())
                        if cls == 0:
                            person_boxes.append(box)
                    except Exception as e:
                        st.write(f"Box processing error: {e}")

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

                    if prediction != 0:  # Alert if not normal driving
                        with open("preview.mp3", "rb") as f:
                            audio_bytes = f.read()
                        audio_b64 = base64.b64encode(audio_bytes).decode()
                        audio_html = f"""
                        <audio autoplay>
                            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                        </audio>
                        """
                        st.markdown(audio_html, unsafe_allow_html=True)
            else:
                st.write("No person detected.")
                
    elif detection_mode == "Video Detection":
        st.write("### Video Detection")
        st.subheader("Upload a Video for Driver Behavior Analysis")
        video_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
        if video_file is not None:
            # Save the video to a temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            frames = []
            detection_log = []  # List to store each frame's detection activity
            frame_count = 0

            with st.spinner("Processing video, please wait..."):
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    annotated_frame, pred = process_frame(frame)
                    frames.append(annotated_frame)
                    detection_log.append({"Frame": frame_count, "Activity": pred})
            cap.release()

            if frames:
                summarized_log = summarize_detection_log(detection_log)
                overall_pred = Counter([entry["Activity"] for entry in detection_log]).most_common(1)[0][0]
                height, width, _ = frames[0].shape
                output_path = os.path.join(tempfile.gettempdir(), "output_video.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, 20, (width, height))
                for f in frames:
                    out.write(f)
                out.release()
                st.success(f"Video processed: {frame_count} frames.")
                st.write(f"### Overall Predicted Activity for Video: {overall_pred}")
                
                # Read the video file as bytes and display it
                with open(output_path, "rb") as f:
                    video_bytes = f.read()
                st.video(video_bytes)
                
                st.write("### Detection Log Summary:")
                st.table(summarized_log)
            else:
                st.error("No frames were processed.")
            os.unlink(tfile.name)

# ---------------- Sidebar: Class Labels ----------------
st.sidebar.subheader("Class Labels for Driver Behavior Classification:")
for key, value in class_labels.items():
    st.sidebar.write(f"**{key}: {value}**")
