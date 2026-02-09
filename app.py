import streamlit as st
from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np
import base64
import tempfile
import os

st.set_page_config(layout="wide")
st.title("🕳️ Pothole Detection + Segmentation (Image + Video)")
st.write("Bounding Boxes, Segmentation Overlay, and Binary Mask")

# ----------------------------
# Roboflow Client
# ----------------------------
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="7l5BKkxbenEWpBCBPtSw"
)

# ----------------------------
# Input Selector
# ----------------------------
file_type = st.radio("Select Input Type", ["Image", "Video"])

if file_type == "Image":
    uploaded_file = st.file_uploader("Upload road image", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.file_uploader("Upload road video", type=["mp4", "mov", "avi", "mkv"])


# ==========================================================
# FUNCTION TO EXTRACT PREDICTIONS SAFELY
# ==========================================================
def extract_predictions(result):
    if isinstance(result, list) and len(result) > 0:
        result = result[0]

    if isinstance(result, dict):
        if "predictions" in result:
            if isinstance(result["predictions"], dict) and "predictions" in result["predictions"]:
                return result["predictions"]["predictions"]

            if isinstance(result["predictions"], list):
                return result["predictions"]

    return []


# ==========================================================
# PROCESS FRAME FUNCTION
# ==========================================================
def process_frame(frame, predictions):
    h, w, _ = frame.shape

    # 1) Bounding Boxes
    bbox_image = frame.copy()
    for p in predictions:
        if all(k in p for k in ["x", "y", "width", "height"]):
            x1 = int(p["x"] - p["width"] / 2)
            y1 = int(p["y"] - p["height"] / 2)
            x2 = int(p["x"] + p["width"] / 2)
            y2 = int(p["y"] + p["height"] / 2)
            cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 2) Segmentation Overlay
    seg_overlay = frame.copy()
    for obj in predictions:
        if "points" in obj:
            pts = np.array([[int(pt["x"]), int(pt["y"])] for pt in obj["points"]], dtype=np.int32)
            cv2.fillPoly(seg_overlay, [pts], (0, 255, 0))

    seg_overlay = cv2.addWeighted(frame, 0.6, seg_overlay, 0.4, 0)

    # 3) Binary Mask
    binary_mask = np.zeros((h, w), dtype=np.uint8)
    for obj in predictions:
        if "points" in obj:
            pts = np.array([[int(pt["x"]), int(pt["y"])] for pt in obj["points"]], dtype=np.int32)
            cv2.fillPoly(binary_mask, [pts], 255)

    binary_mask_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    return bbox_image, seg_overlay, binary_mask_bgr, binary_mask


# ==========================================================
# IMAGE MODE
# ==========================================================
if uploaded_file and file_type == "Image":

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📤 Uploaded Image")
        st.image(uploaded_file, use_container_width=True)

    image_bytes = uploaded_file.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    with st.spinner("Running detection + segmentation..."):
        result = client.run_workflow(
            workspace_name="project1-mflte",
            workflow_id="detect-count-and-visualize-2",
            images={"image": image_base64}
        )

    predictions = extract_predictions(result)

    img_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    bbox_img, overlay_img, mask_bgr, mask_gray = process_frame(image, predictions)

    bbox_rgb = cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("📦 Bounding Boxes")
        st.image(bbox_rgb, use_container_width=True)

    with col3:
        st.subheader("🟩 Segmentation Overlay")
        st.image(overlay_rgb, use_container_width=True)

    st.divider()
    st.subheader("⚫⚪ Binary Segmentation Mask")
    st.image(mask_gray, clamp=True)
    st.caption("White = pothole, Black = background")

    pothole_count = len(predictions)
    st.success(f"🕳️ Total potholes detected: {pothole_count}")

    cv2.imwrite("binary_mask.png", mask_gray)

    st.download_button(
        "⬇️ Download Binary Mask",
        open("binary_mask.png", "rb"),
        file_name="pothole_binary_mask.png"
    )

    with st.expander("Show raw output"):
        st.write(result)


# ==========================================================
# VIDEO MODE
# ==========================================================
if uploaded_file and file_type == "Video":

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if input_fps == 0:
        st.error("❌ Could not read video FPS. Upload another video.")
        st.stop()

    st.info(f"🎥 Input FPS: {input_fps:.2f} | Resolution: {width}x{height}")
    st.warning("Processing only 1 frame per second (1 FPS output)")

    skip_frames = int(input_fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out_original = "original_video.mp4"
    out_bbox = "bbox_video.mp4"
    out_overlay = "overlay_video.mp4"
    out_mask = "mask_video.mp4"

    writer_original = cv2.VideoWriter(out_original, fourcc, 1, (width, height))
    writer_bbox = cv2.VideoWriter(out_bbox, fourcc, 1, (width, height))
    writer_overlay = cv2.VideoWriter(out_overlay, fourcc, 1, (width, height))
    writer_mask = cv2.VideoWriter(out_mask, fourcc, 1, (width, height))

    frame_count = 0
    processed_frames = 0
    progress = st.progress(0)

    with st.spinner("Processing video frames..."):

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % skip_frames == 0:
                processed_frames += 1

                _, buffer = cv2.imencode(".jpg", frame)
                frame_base64 = base64.b64encode(buffer).decode("utf-8")

                result = client.run_workflow(
                    workspace_name="project1-mflte",
                    workflow_id="detect-count-and-visualize-2",
                    images={"image": frame_base64}
                )

                predictions = extract_predictions(result)

                bbox_img, overlay_img, mask_bgr, _ = process_frame(frame, predictions)

                writer_original.write(frame)
                writer_bbox.write(bbox_img)
                writer_overlay.write(overlay_img)
                writer_mask.write(mask_bgr)

            frame_count += 1
            progress.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    writer_original.release()
    writer_bbox.release()
    writer_overlay.release()
    writer_mask.release()

    st.success(f"✅ Video Processing Completed! Processed Frames: {processed_frames}")

    st.divider()
    st.subheader("🎬 Output Videos (1 FPS)")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Video")
        st.video(out_original)

        st.subheader("Bounding Boxes Video")
        st.video(out_bbox)

    with col2:
        st.subheader("Segmentation Overlay Video")
        st.video(out_overlay)

        st.subheader("Binary Mask Video")
        st.video(out_mask)

    st.divider()

    st.download_button("⬇️ Download Original Video", open(out_original, "rb"), file_name="original_video.mp4")
    st.download_button("⬇️ Download Bounding Boxes Video", open(out_bbox, "rb"), file_name="bbox_video.mp4")
    st.download_button("⬇️ Download Segmentation Overlay Video", open(out_overlay, "rb"), file_name="overlay_video.mp4")
    st.download_button("⬇️ Download Binary Mask Video", open(out_mask, "rb"), file_name="mask_video.mp4")

    os.remove(tfile.name)
