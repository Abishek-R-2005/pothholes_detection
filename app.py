from typing import NamedTuple
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# ------------------------------------------------
# Streamlit configuration
# ------------------------------------------------
st.set_page_config(
    page_title="Pothole Detection - Image",
    page_icon="🕳️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------
# LOCAL MODEL PATH (YOUR PATH)
# ------------------------------------------------
MODEL_PATH = Path("/Users/abishekr/Downloads/YOLOv8_Small_RDD.pt")

if not MODEL_PATH.exists():
    st.error("❌ Model file not found at the given path")
    st.stop()

# ------------------------------------------------
# Load YOLO model (cached)
# ------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))

model = load_model()

# ------------------------------------------------
# POTHOLE CLASS CONFIG
# ------------------------------------------------
POTHOLE_CLASS_ID = 3   # 🔴 change ONLY if your dataset differs
POTHOLE_LABEL = "Pothole"

class Detection(NamedTuple):
    score: float
    box: np.ndarray

# ------------------------------------------------
# UI
# ------------------------------------------------
st.title("🕳️ Pothole Detection (Image Only)")
st.write(
    "This app detects **ONLY potholes** from a road image using a locally stored YOLOv8 model."
)

image_file = st.file_uploader(
    "📤 Upload Road Image",
    type=["png", "jpg", "jpeg"]
)

confidence = st.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=1.0,
    value=0.4,
    step=0.05
)

st.caption(
    "⬇️ Lower confidence if potholes are missed. Increase it if false potholes appear."
)

# ------------------------------------------------
# INFERENCE
# ------------------------------------------------
if image_file is not None:

    image = Image.open(image_file).convert("RGB")
    img = np.array(image)

    h, w = img.shape[:2]

    img_resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)

    with st.spinner("Detecting potholes..."):
        results = model.predict(
            img_resized,
            conf=confidence,
            verbose=False
        )

    pothole_detections = []

    # Extract ONLY potholes
    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            if int(box.cls) == POTHOLE_CLASS_ID:
                pothole_detections.append(
                    Detection(
                        score=float(box.conf),
                        box=box.xyxy[0].cpu().numpy().astype(int)
                    )
                )

    # ------------------------------------------------
    # Draw ONLY pothole boxes manually (OpenCV style)
    # ------------------------------------------------
    output_img = img.copy()

    scale_x = w / 640
    scale_y = h / 640

    for det in pothole_detections:
        x1, y1, x2, y2 = det.box

        # Rescale box back to original image
        x1 = int(x1 * scale_x)
        x2 = int(x2 * scale_x)
        y1 = int(y1 * scale_y)
        y2 = int(y2 * scale_y)

        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        label = f"{POTHOLE_LABEL} {det.score:.2f}"
        cv2.putText(
            output_img,
            label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # ------------------------------------------------
    # DISPLAY
    # ------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(img, use_column_width=True)

    with col2:
        st.subheader("Pothole Detection")
        st.image(output_img, use_column_width=True)

    st.success(f"🕳️ Total potholes detected: {len(pothole_detections)}")

    # ------------------------------------------------
    # DOWNLOAD OUTPUT
    # ------------------------------------------------
    buffer = BytesIO()
    Image.fromarray(output_img).save(buffer, format="PNG")
    buffer.seek(0)

    st.download_button(
        label="⬇️ Download Pothole Detection Image",
        data=buffer,
        file_name="pothole_detection.png",
        mime="image/png"
    )
