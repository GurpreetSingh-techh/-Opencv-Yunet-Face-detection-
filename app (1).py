
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import io

st.set_page_config(
    page_title="Face Detection Comparison",
    page_icon="👤",
    layout="wide"
)

st.title("Face Detection Using OpenCV and YuNet")

st.write(
    "Compare Haar Cascade and YuNet face detection "
    "on the same image."
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HAAR_PATH = os.path.join(
    BASE_DIR,
    "models",
    "haarcascade_frontalface_default.xml"
)

YUNET_PATH = os.path.join(
    BASE_DIR,
    "models",
    "face_detection_yunet_2023mar.onnx"
)


# LOAD HAAR CASCADE

@st.cache_resource
def load_haar():

    model = cv2.CascadeClassifier(HAAR_PATH)

    if model.empty():
        raise RuntimeError("Haar model could not be loaded.")

    return model


# LOAD YUNET

@st.cache_resource
def load_yunet():

    return cv2.FaceDetectorYN.create(
        YUNET_PATH,
        "",
        (320, 320),
        0.8,
        0.3,
        5000
    )


if not os.path.exists(HAAR_PATH) or not os.path.exists(YUNET_PATH):

    st.error("Model files are missing!")

    st.stop()


haar_model = load_haar()
yunet_model = load_yunet()

st.success("Both face detection models loaded successfully!")


# IMAGE UPLOAD

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    image_rgb = np.array(image)

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )


    # HAAR CASCADE DETECTION

    faces_haar = haar_model.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    haar_output = image_bgr.copy()

    for (x, y, w, h) in faces_haar:

        cv2.rectangle(
            haar_output,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )


    # YUNET DETECTION

    height, width = image_bgr.shape[:2]

    yunet_model.setInputSize((width, height))

    _, faces_yunet = yunet_model.detect(image_bgr)

    yunet_output = image_bgr.copy()

    yunet_count = 0

    if faces_yunet is not None:

        yunet_count = len(faces_yunet)

        for face in faces_yunet:

            x, y, w, h = face[:4].astype(int)

            confidence = face[14]

            cv2.rectangle(
                yunet_output,
                (x, y),
                (x+w, y+h),
                (255, 0, 0),
                2
            )

            cv2.putText(
                yunet_output,
                f"{confidence:.2f}",
                (x, max(y-10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )


    # DISPLAY RESULTS

    st.subheader("Detection Results")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("OpenCV Haar Cascade")

        st.image(
            cv2.cvtColor(haar_output, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

        st.metric(
            "Faces Detected",
            len(faces_haar)
        )


    with col2:

        st.subheader("YuNet Face Detection")

        st.image(
            cv2.cvtColor(yunet_output, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

        st.metric(
            "Faces Detected",
            yunet_count
        )


    # DOWNLOAD RESULTS

    st.subheader("Download Results")

    col1, col2 = st.columns(2)

    def convert_image(img):

        buffer = io.BytesIO()

        Image.fromarray(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ).save(buffer, format="JPEG")

        return buffer.getvalue()


    with col1:

        st.download_button(
            "Download Haar Result",
            convert_image(haar_output),
            "haar_detection.jpg",
            "image/jpeg"
        )


    with col2:

        st.download_button(
            "Download YuNet Result",
            convert_image(yunet_output),
            "yunet_detection.jpg",
            "image/jpeg"
        )
