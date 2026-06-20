import streamlit as st
import requests
from PIL import Image
import io

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Crop Disease Diagnosis",
    page_icon="🌿",
    layout="wide"
)

import os
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# ── Header ────────────────────────────────────────────────
st.title("🌿 Crop Disease Diagnosis System")
st.markdown("Upload a leaf image to detect diseases and get treatment recommendations.")
st.divider()

# ── Layout ────────────────────────────────────────────────
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📷 Upload Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)


        if st.button("🔍 Diagnose", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                try:
                    files    = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")}
                    response = requests.post(f"{API_URL}/predict", files=files)
                    result   = response.json()

                    # Store in session state
                    st.session_state["result"] = result

                except Exception as e:
                    st.error(f"Error: {e}")

with col2:
    st.subheader("🔬 Diagnosis Results")

    if "result" in st.session_state:
        result = st.session_state["result"]

        # Disease name
        disease_raw   = result["disease"]
        disease_clean = disease_raw.replace("___", " → ").replace("_", " ")
        confidence    = result["confidence"]

        # Confidence color
        if confidence >= 70:
            conf_color = "🟢"
        elif confidence >= 40:
            conf_color = "🟡"
        else:
            conf_color = "🔴"

        st.markdown(f"### {conf_color} {disease_clean}")
        st.markdown(f"**Confidence:** {confidence}%")
        st.divider()

        # Treatment info
        st.markdown("### 💊 Treatment & Recommendations")
        st.markdown(result["treatment"])

        st.divider()

        # ── Q&A Section ───────────────────────────────────
        st.subheader("💬 Ask a Follow-up Question")
        question = st.text_input(
            "Ask anything about this disease...",
            placeholder="e.g. What organic treatment works best?"
        )

        if st.button("Ask", use_container_width=True) and question:
            with st.spinner("Getting answer..."):
                try:
                    response = requests.post(
                        f"{API_URL}/ask",
                        params={
                            "disease"  : disease_raw,
                            "question" : question
                        }
                    )
                    answer = response.json()["answer"]
                    st.markdown("**Answer:**")
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("Upload an image and click Diagnose to see results here.")