import streamlit as st
from inference_sdk import InferenceHTTPClient
import base64

st.set_page_config(layout="wide")
st.title("🕳️ Pothole Detection App")

uploaded_file = st.file_uploader(
    "Upload road image (wet or dry)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Show uploaded image
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📤 Uploaded Image")
        st.image(uploaded_file, width=400)

    # ---- 🔑 KEY FIX HERE ----
    image_bytes = uploaded_file.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="7l5BKkxbenEWpBCBPtSw"
    )

    with st.spinner("Detecting potholes..."):
        result = client.run_workflow(
            workspace_name="project1-mflte",
            workflow_id="find-potholes",
            images={
                "image": image_base64   # ✅ THIS FIXES EVERYTHING
            }
        )

    output = result[0]

    # Output image
    output_image_base64 = output["visualization"]
    output_image_bytes = base64.b64decode(output_image_base64)

    with col2:
        st.subheader("📊 Detection Output")
        st.image(output_image_bytes, width=400)

        pothole_count = len(output["predictions"]["predictions"])
        st.success(f"🕳️ **Total potholes detected: {pothole_count}**")

    st.subheader("🧾 Full Predictions JSON")
    st.json(output["predictions"])
