import streamlit as st
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
MODEL_PATH = "distracted-16-0.99.keras"
model = load_model(MODEL_PATH)

# Define class labels
class_labels = [
    "Normal driving",
    "Texting - right",
    "Talking on the phone - right",
    "Texting - left",
    "Talking on the phone - left",
    "Operating the radio",
    "Drinking",
    "Reaching behind",
    "Hair and makeup",
    "Talking to passenger",
]

# Streamlit title
st.title("Driver Distraction Detection")
st.write("Use your webcam for real-time driver distraction detection.")

# Button to start the webcam feed
start_webcam = st.button("Start Webcam")

if start_webcam:
    # Start video capture
    video_capture = cv2.VideoCapture(0)  # Use 0 for the default webcam

    # Stream the video
    st_frame = st.empty()  # Placeholder for the video frame

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            st.write("Failed to capture video. Stopping...")
            break

        # Preprocess the frame for the model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        resized_frame = cv2.resize(frame_rgb, (64, 64))  # Resize to match model input
        normalized_frame = np.array(resized_frame) / 255.0  # Normalize
        input_frame = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension

        # Predict using the model
        predictions = model.predict(input_frame)
        predicted_class = np.argmax(predictions[0])
        predicted_label = class_labels[predicted_class]

        # Add the prediction to the frame
        cv2.putText(
            frame,
            f"Prediction: {predicted_label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Display the frame in Streamlit
        st_frame.image(frame, channels="BGR")

    video_capture.release()
    cv2.destroyAllWindows()
