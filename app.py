import streamlit as st
from inference_sdk import InferenceHTTPClient
import base64
import numpy as np
import cv2
import json

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(
    page_title="Pothole Detection",
    page_icon="🕳️",
    layout="wide"
)

st.title("🕳️ Pothole Detection App")
st.write("Upload a road image (wet or dry) and detect potholes automatically.")

uploaded_file = st.file_uploader(
    "Upload Road Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image_bytes = uploaded_file.read()

    # ----------------------------
    # Two-column layout
    # ----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image_bytes, width=500)

    # ----------------------------
    # Roboflow Client
    # ----------------------------
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="7l5BKkxbenEWpBCBPtSw"
    )

    with st.spinner("Detecting potholes..."):
        result = client.run_workflow(
        workspace_name="project1-mflte",
        workflow_id="find-potholes",
        images={"image": uploaded_file}
    )




    # ----------------------------
    # Decode visualization image
    # ----------------------------
    viz_base64 = result[0]["visualization"]
    decoded_bytes = base64.b64decode(viz_base64)
    image_np = np.frombuffer(decoded_bytes, dtype=np.uint8)
    output_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    # ----------------------------
    # Predictions & count
    # ----------------------------
    predictions_data = result[0]["predictions"]
    pothole_count = len(predictions_data.get("predictions", []))

    with col2:
        st.subheader("Detected Potholes")
        st.image(output_image_rgb, width=500)
        st.markdown(
            f"### 🕳️ Total potholes detected: **{pothole_count}**"
        )

    # ----------------------------
    # Full JSON Output (no scroll)
    # ----------------------------
    st.divider()
    st.subheader("Full Predictions JSON")

    full_json = json.dumps(predictions_data, indent=4)
    st.code(full_json, language="json")
