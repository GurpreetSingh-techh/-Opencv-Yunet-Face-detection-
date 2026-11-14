
# OpenCV Haar Cascade & YuNet Face Detection

A Computer Vision project that implements two independent face detection algorithms: OpenCV Haar Cascade and YuNet (Deep Learning).

The application allows users to upload an image and detect faces using both algorithms independently.

## Features

- Upload images (JPG, JPEG, PNG).
- Face detection using OpenCV Haar Cascade.
- Face detection using YuNet pretrained ONNX model.
- Independent face detection using both algorithms.
- Draw bounding boxes around detected faces.
- Display the number of detected faces.
- Display YuNet confidence scores.
- Download detected images.
- Compare both detection methods.

## Technologies Used

- Python
- OpenCV
- Haar Cascade
- YuNet
- ONNX
- NumPy
- Pillow
- Streamlit

## Project Structure

```text
OpenCV-HaarCascade-YuNet-Face-Detection/
│
├── app.py
├── requirements.txt
├── README.md
│
└── models/
    ├── haarcascade_frontalface_default.xml
    └── face_detection_yunet_2023mar.onnx
