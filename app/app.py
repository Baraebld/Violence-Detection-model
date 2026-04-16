# app/app.py
import sys
from pathlib import Path
# =========================
# Project root for imports
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
import streamlit as st
import torch
import tempfile
from src.video_utils import load_video_frames
from src.load_model import load_trained_model

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="🔥 Violence Detection in Videos",
    layout="wide"
)

# =========================
# App Title & Description
# =========================
st.title("🔥 Violence Detection in Videos")
st.markdown(
    """
    Upload a video and get a **real-time prediction** if it is violent or non-violent.
    Wait a few seconds for model prediction,Probabilities are shown as progress bars
    """
)
# =========================
# Load model
# =========================
with st.spinner("Loading model..."):
    model, device = load_trained_model()

# =========================
# Video Upload
# =========================
uploaded_file = st.file_uploader(
    "Upload a video file",
    type=["mp4", "avi", "mov", "mkv"],
    help="Supported formats: MP4, AVI, MOV, MKV"
)

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Display video and prediction side by side
    with st.container():
        col1, col2 = st.columns([1.3, 1])
        
        with col1:
            st.subheader("Video Preview")
            st.markdown("<div style='max-width:300px'>", unsafe_allow_html=True)
            st.video(video_path)
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.subheader("Prediction")
            with st.spinner("Analyzing video..."):
                try:
                    # Load frames and prepare tensor
                    video_tensor = load_video_frames(video_path).unsqueeze(0).to(device)

                    # Forward pass
                    with torch.no_grad():
                        output = model(video_tensor)
                        prob = torch.sigmoid(output).item()
                        violence_prob = prob * 100
                        non_violence_prob = 100 - violence_prob

                    # Display probabilities
                    st.markdown("**Violence Probability**")
                    st.progress(violence_prob / 100)
                    st.markdown("**Non-Violence Probability**")
                    st.progress(non_violence_prob / 100)

                    # Highlight main prediction
                    if violence_prob > 50:
                        st.error("⚠️ This video is  violent.")
                    else:
                        st.success("✅ This video is  non-violent.")

                except Exception as e:
                    st.error(f"Error processing video: {e}")

    # Optional: cleanup temp file
    # Path(video_path).unlink(missing_ok=True)

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown(
    "Developed with ❤️ by RIHAM | [GitHub Repo](https://github.com/riham-esi/Violence-Detection-video.git)"
)