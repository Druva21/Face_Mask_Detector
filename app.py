import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import time
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# ======================
# Page Config
# ======================
st.set_page_config(page_title='Face Mask Detector', page_icon='😷', layout='centered', initial_sidebar_state='expanded')

# ======================
# Utility Functions
# ======================
@st.cache_resource
def load_models():
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)
    model = load_model("mask_detector.model")
    return net, model


def detect_mask_from_array(image_bgr):
    """
    Runs mask detection on a BGR image array and returns processed image + prediction text.
    """
    net, model = load_models()

    image = image_bgr.copy()
    if image is None or image.size == 0:
        return None, "Error: Could not read image."

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    predictions = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = image[startY:endY, startX:endX]
            if face.size == 0: 
                continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = model.predict(face)[0]
            label = "Mask" if mask > withoutMask else "No Mask"
            prob = max(mask, withoutMask) * 100
            predictions.append(f"{label} ({prob:.2f}%)")

            # Draw on image
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            text = f"{label}: {prob:.2f}%"
            cv2.putText(image, text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    if not predictions:
        predictions = ["No Face Detected"]

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb), predictions


def detect_mask(image_path):
    """
    Backwards-compatible wrapper: reads an image from disk and calls array-based detector.
    """
    image = cv2.imread(image_path)
    return detect_mask_from_array(image)


# ======================
# Streamlit App
# ======================
def mask_detection():
    st.markdown('<h1 style="text-align:center;">😷 Face Mask Detection</h1>', unsafe_allow_html=True)
    activities = ["Image Upload", "Webcam", "Live Webcam (Auto)"]
    choice = st.sidebar.selectbox("Choose Detection Mode:", activities)

    if choice == 'Image Upload':
        st.subheader("Upload an Image")
        image_file = st.file_uploader("Choose a JPG/PNG file", type=['jpg', 'png'])
        if image_file is not None:
            # Save uploaded image
            img_path = "./images/uploaded.jpg"
            with open(img_path, "wb") as f:
                f.write(image_file.getbuffer())
            uploaded_image = Image.open(image_file)
            placeholder = st.empty()
            placeholder.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

            if st.button("Done"):
                processed_image, predictions = detect_mask(img_path)
                placeholder.image(processed_image, caption="Result", use_column_width=True)
                st.success("Prediction: " + ", ".join(predictions))

    elif choice == 'Webcam':
        st.subheader("Take a Photo")
        img_file_buffer = st.camera_input("Capture from Webcam")
        if img_file_buffer is not None:
            img_path = "./images/webcam.jpg"
            with open(img_path, "wb") as f:
                f.write(img_file_buffer.getbuffer())
            webcam_image = Image.open(img_file_buffer)
            placeholder = st.empty()
            placeholder.image(webcam_image, caption="Captured Photo", use_column_width=True)

            if st.button("Done"):
                processed_image, predictions = detect_mask(img_path)
                placeholder.image(processed_image, caption="Result", use_column_width=True)
                st.success("Prediction: " + ", ".join(predictions))

    elif choice == 'Live Webcam (Auto)':
        st.subheader("Live Detection (Auto)")
        start = st.checkbox("Start camera")
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        if start:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open webcam.")
            else:
                # Run until checkbox is unchecked
                while st.session_state.get("live_running", True) and start:
                    ret, frame = cap.read()
                    if not ret:
                        status_placeholder.error("Failed to read frame from webcam.")
                        break
                    processed_image, predictions = detect_mask_from_array(frame)
                    frame_placeholder.image(processed_image, caption=", ".join(predictions) if isinstance(predictions, list) else str(predictions), use_column_width=True)
                    status_placeholder.info("Prediction: " + (", ".join(predictions) if isinstance(predictions, list) else str(predictions)))
                    time.sleep(0.05)
            cap.release()
            cv2.destroyAllWindows()


# Run App
mask_detection()
