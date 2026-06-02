"""
app.py — Streamlit Web Application for Digit Recognition
==========================================================
Author : Anas Dhafer Alzahrani
GitHub : https://github.com/Rflx9
Email  : anaas123190@gmail.com
----------------------------------------------------------
Interactive web app that lets users:
  1. Draw a digit on a canvas
  2. Upload an image of a handwritten digit
  3. Get real-time predictions from the trained CNN model

Usage:
    streamlit run app.py
"""

import os
import io
import numpy as np
from PIL import Image, ImageOps, ImageFilter

import streamlit as st
import tensorflow as tf

# ──────────────────────────────────────────────
# Page Configuration (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Digit Recognizer · MNIST CNN",
    page_icon="🔢",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
MODEL_PATH   = "mnist_cnn_model.keras"
CLASS_NAMES  = list(range(10))    # [0, 1, 2, ..., 9]
IMG_SIZE     = 28                 # MNIST images are 28×28 pixels


# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — Dark, modern UI
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0F0F1A;
    color: #E8E8F0;
}

/* ── Header ── */
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00D4FF 0%, #7B68EE 50%, #FF6B6B 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    line-height: 1.2;
    margin-bottom: 0.2rem;
}
.hero-subtitle {
    text-align: center;
    color: #888899;
    font-size: 1rem;
    margin-bottom: 2rem;
}

/* ── Cards ── */
.card {
    background: #1A1A2E;
    border: 1px solid #2A2A45;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.2rem;
}

/* ── Prediction display ── */
.pred-box {
    background: linear-gradient(135deg, #1A1A2E 0%, #0D0D20 100%);
    border: 2px solid #00D4FF;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 30px rgba(0, 212, 255, 0.15);
}
.pred-digit {
    font-size: 5rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, #00D4FF, #7B68EE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.pred-confidence {
    font-size: 1.3rem;
    color: #00FF88;
    font-weight: 600;
    margin-top: 0.5rem;
}
.pred-label {
    color: #888899;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.3rem;
}

/* ── Probability bar ── */
.prob-bar-container {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 4px 0;
}
.prob-label {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: #E8E8F0;
    width: 20px;
    text-align: right;
}
.prob-bar-bg {
    flex: 1;
    height: 10px;
    background: #2A2A45;
    border-radius: 5px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 5px;
    transition: width 0.4s ease;
}
.prob-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #888899;
    width: 48px;
    text-align: right;
}

/* ── Streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #00D4FF, #7B68EE);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem;
    cursor: pointer;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

[data-testid="stFileUploader"] {
    background: #1A1A2E;
    border: 2px dashed #2A2A45;
    border-radius: 12px;
    padding: 1rem;
}

.stTabs [data-baseweb="tab-list"] {
    background: #1A1A2E;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #888899;
    border-radius: 8px;
    font-weight: 600;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: #2A2A45 !important;
    color: #00D4FF !important;
}

.stAlert { border-radius: 10px; }

hr { border-color: #2A2A45; }

.info-chip {
    display: inline-block;
    background: #1A1A2E;
    border: 1px solid #2A2A45;
    color: #888899;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.8rem;
    margin: 3px;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Model Loading (cached so it only loads once per session)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained model from disk (cached for performance)."""
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


# ──────────────────────────────────────────────────────────────────────────────
# Image Preprocessing
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess a PIL Image to match MNIST input format.

    Steps:
      1. Convert to grayscale
      2. Invert if background is light (MNIST uses white digits on black)
      3. Resize to 28×28
      4. Normalize pixel values to [0, 1]
      5. Add batch and channel dimensions

    Args:
        image: PIL Image (any size, any mode).

    Returns:
        Numpy array of shape (1, 28, 28, 1) ready for model inference.
    """
    # Convert to grayscale
    img = image.convert("L")

    # Check if image has a light background (mean pixel > 127)
    # MNIST has white-on-black, so we invert if needed
    img_array = np.array(img)
    if img_array.mean() > 127:
        img = ImageOps.invert(img)

    # Gentle smoothing to reduce noise from drawing/upload artifacts
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))

    # Resize to 28×28 using high-quality downsampling
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)

    # Convert to numpy and normalize
    img_array = np.array(img, dtype="float32") / 255.0

    # Add batch dimension (1,) and channel dimension (1)
    # Shape: (28, 28) → (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=(0, -1))

    return img_array


def predict(model, img_array: np.ndarray):
    """
    Run the model and return predicted digit and confidence scores.

    Args:
        model: Loaded Keras model.
        img_array: Preprocessed image array of shape (1, 28, 28, 1).

    Returns:
        predicted_digit (int): The digit with highest probability.
        probabilities (list):  Confidence scores for each digit 0–9.
    """
    # Get softmax probabilities (shape: (1, 10))
    probs = model.predict(img_array, verbose=0)[0]

    # Pick the digit with the highest probability
    predicted_digit = int(np.argmax(probs))

    return predicted_digit, probs.tolist()


# ──────────────────────────────────────────────────────────────────────────────
# UI Components
# ──────────────────────────────────────────────────────────────────────────────
def render_prediction_card(digit: int, confidence: float):
    """Render the big prediction result card."""
    st.markdown(f"""
    <div class="pred-box">
        <div class="pred-label">Predicted Digit</div>
        <div class="pred-digit">{digit}</div>
        <div class="pred-confidence">↑ {confidence*100:.1f}% confidence</div>
    </div>
    """, unsafe_allow_html=True)


def render_probability_bars(probs: list):
    """Render a styled horizontal bar chart for class probabilities."""
    st.markdown("**Probability Distribution**")

    max_prob = max(probs)
    bars_html = ""

    for i, p in enumerate(probs):
        # Color: bright for top prediction, muted for others
        if p == max_prob:
            color = "linear-gradient(90deg, #00D4FF, #7B68EE)"
        elif p > 0.05:
            color = "linear-gradient(90deg, #FF6B6B, #FF8E53)"
        else:
            color = "#2A2A45"

        width_pct = p * 100
        bars_html += f"""
        <div class="prob-bar-container">
            <span class="prob-label">{i}</span>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{width_pct:.1f}%; background:{color};"></div>
            </div>
            <span class="prob-value">{p*100:.1f}%</span>
        </div>"""

    st.markdown(
        f'<div class="card">{bars_html}</div>',
        unsafe_allow_html=True
    )


def render_image_preview(img_array: np.ndarray):
    """Show the preprocessed 28×28 image fed into the model."""
    preview = (img_array.squeeze() * 255).astype("uint8")
    preview_img = Image.fromarray(preview, mode="L")
    preview_img = preview_img.resize((140, 140), Image.Resampling.NEAREST)
    st.image(preview_img, caption="📐 28×28 input to model", width=140)


# ──────────────────────────────────────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # ── Hero header ─────────────────────────────────────────────────────────
    st.markdown(
        '<div class="hero-title">🔢 Digit Recognizer</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="hero-subtitle">'
        'Handwritten Digit Recognition · MNIST CNN · TensorFlow / Keras'
        '</div>',
        unsafe_allow_html=True
    )

    # ── Load model ──────────────────────────────────────────────────────────
    model = load_model()

    if model is None:
        st.error(
            "⚠️ **Model not found!**\n\n"
            "Please train the model first by running:\n"
            "```bash\npython train.py\n```",
            icon="🚫"
        )
        st.stop()

    # ── Input tabs ──────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📁 Upload Image", "ℹ️ About"])

    # ────────────────────────────────────────────────────────────────────────
    # Tab 1: Upload Image
    # ────────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("#### Upload a handwritten digit image")
        st.markdown(
            "<small style='color:#888899'>Supports PNG, JPG, JPEG, BMP, GIF "
            "— any size, color or grayscale</small>",
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg", "bmp", "gif"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            # Load image from upload
            image = Image.open(io.BytesIO(uploaded_file.read()))

            col_img, col_arrow, col_result = st.columns([2, 1, 3])

            with col_img:
                st.markdown("**Original Image**")
                display_img = image.copy()
                display_img.thumbnail((200, 200))
                st.image(display_img, use_column_width=False)

            with col_arrow:
                st.markdown("<br><br><br>", unsafe_allow_html=True)
                st.markdown(
                    "<div style='text-align:center;font-size:2rem;'>→</div>",
                    unsafe_allow_html=True
                )

            with col_result:
                # Preprocess and predict
                with st.spinner("Running CNN inference..."):
                    img_array = preprocess_image(image)
                    digit, probs = predict(model, img_array)

                # Show result
                render_prediction_card(digit, probs[digit])
                st.markdown("<br>", unsafe_allow_html=True)
                render_image_preview(img_array)

            st.divider()
            render_probability_bars(probs)

        else:
            st.markdown("""
            <div class="card" style="text-align:center; padding: 3rem;">
                <div style="font-size:3rem;">🖼️</div>
                <div style="color:#888899; margin-top:1rem;">
                    Upload a PNG, JPG, or BMP image of a handwritten digit (0–9).<br>
                    <small>The app works best with clear, centered digits on a clean background.</small>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────────
    # Tab 2: About
    # ────────────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("""
        <div class="card">
        <h4 style="color:#00D4FF;">📖 About This Project</h4>
        <p>
        This app uses a <strong>Convolutional Neural Network (CNN)</strong> trained on the 
        <strong>MNIST dataset</strong> — 70,000 handwritten digit images — to classify 
        digits 0 through 9 in real time.
        </p>
        <h5 style="color:#7B68EE; margin-top:1.5rem;">🏗️ Model Architecture</h5>
        <ul>
            <li><strong>3× Conv2D layers</strong> — Extract features (edges, curves, loops)</li>
            <li><strong>2× MaxPooling2D</strong> — Downsample and reduce overfitting</li>
            <li><strong>Dropout (50%)</strong> — Regularization against overfitting</li>
            <li><strong>Dense (128)</strong> — High-level feature combination</li>
            <li><strong>Softmax output (10)</strong> — Probability over 10 classes</li>
        </ul>
        <h5 style="color:#7B68EE; margin-top:1.5rem;">📊 Performance</h5>
        <ul>
            <li>Test Accuracy: ~<strong>99.2%</strong></li>
            <li>Training samples: <strong>60,000</strong></li>
            <li>Test samples: <strong>10,000</strong></li>
        </ul>
        <h5 style="color:#7B68EE; margin-top:1.5rem;">🛠️ Tech Stack</h5>
        <span class="info-chip">Python 3.10+</span>
        <span class="info-chip">TensorFlow / Keras</span>
        <span class="info-chip">Streamlit</span>
        <span class="info-chip">NumPy</span>
        <span class="info-chip">Pillow</span>
        <span class="info-chip">Scikit-learn</span>
        <span class="info-chip">Matplotlib</span>
        <span class="info-chip">Seaborn</span>
        <h5 style="color:#7B68EE; margin-top:1.5rem;">👤 Author</h5>
        <p>
        <strong>Anas Dhafer Alzahrani</strong> — Computer Science / AI Student<br>
        <a href="https://github.com/Rflx9" style="color:#00D4FF;">GitHub</a> ·
        <a href="https://linkedin.com/in/anas-zhr26" style="color:#00D4FF;">LinkedIn</a> ·
        <a href="mailto:anaas123190@gmail.com" style="color:#00D4FF;">anaas123190@gmail.com</a>
        </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; color:#555566; font-size:0.8rem; margin-top:3rem; padding-top:1rem; border-top:1px solid #1A1A2E;">
        Handwritten Digit Recognition · MNIST CNN · Built with TensorFlow & Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
