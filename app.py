import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from datetime import datetime

# ---------- Configuration ----------
DEFAULT_MODEL_PATH = "saved_models/efficientnetv2_b0.h5"
LABELS_PATH = "saved_models/class_indices.json"
IMG_SIZE = (224, 224)
DATA_FILE = "scan_data.json"

# ---------- CSS Styling ----------
st.markdown("""
<style>
    .scanner-box {
        border: 3px solid #28a745;
        border-radius: 10px;
        padding: 20px;
        background-color: #f8f9fa;
        margin-bottom: 20px;
    }
    .result-box {
        border: 3px solid #28a745;
        border-radius: 10px;
        padding: 20px;
        background-color: #f8f9fa;
        margin-bottom: 20px;
    }
    .tab-label {
        font-size: 16px;
        font-weight: bold;
        color: #28a745;
        margin-bottom: 15px;
    }
    .prediction-item {
        padding: 8px 12px;
        margin: 5px 0;
        background-color: transparent;
        border-left: 0px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Utilities ----------
@st.cache_resource
def load_model_and_labels(model_path=DEFAULT_MODEL_PATH, labels_path=LABELS_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    with open(labels_path, "r") as f:
        inv_map = json.load(f)
    inv_map = {int(k): v for k, v in inv_map.items()}
    has_rescaling = any("Rescaling" in layer.__class__.__name__ or "Preprocessing" in layer.__class__.__name__ for layer in model.layers)
    return model, inv_map, has_rescaling


def preprocess_image(image: Image.Image, target_size=IMG_SIZE, do_rescale=True):
    image = image.convert("RGB").resize(target_size)
    arr = np.array(image).astype(np.float32)
    if do_rescale:
        arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def load_scan_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {}


def save_scan_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ---------- Initialize Session State ----------
if "scan_data" not in st.session_state:
    st.session_state.scan_data = load_scan_data()
if "scan_triggered" not in st.session_state:
    st.session_state.scan_triggered = False
if "image_to_scan" not in st.session_state:
    st.session_state.image_to_scan = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None


# ---------- Page Config ----------
st.set_page_config(page_title="Fruit & Veg Inventory Scanner", layout="wide")
st.title("🥬 Super Fresh AI")

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["Scanner", "Result", "Data"])

# ========== TAB 1: SCANNER ==========
with tab1:
    st.markdown("<div class='tab-label'>📱 Scanner</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='scanner-box'>", unsafe_allow_html=True)
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="scanner_uploader")
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, use_column_width=True, caption="Selected Image")
        
        if st.button("🔍 Scan Image", use_container_width=True, key="scan_btn"):
            try:
                model, inv_map, has_rescaling = load_model_and_labels()
                x = preprocess_image(img, target_size=IMG_SIZE, do_rescale=not has_rescaling)
                
                with st.spinner("Analyzing..."):
                    preds = model.predict(x, verbose=0)
                probs = preds[0]
                
                top_idx = np.argmax(probs)
                label = inv_map.get(int(top_idx), str(top_idx))
                prob = float(probs[top_idx])
                
                st.session_state.prediction = {"label": label, "prob": prob}
                st.session_state.image_to_scan = img
                st.session_state.scan_triggered = True
                st.success("Image scanned! Go to Result tab to confirm and save.")
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ========== TAB 2: RESULT ==========
with tab2:
    st.markdown("<div class='tab-label'>✅ Result</div>", unsafe_allow_html=True)
    
    if st.session_state.scan_triggered and st.session_state.prediction:
        label = st.session_state.prediction["label"]
        prob = st.session_state.prediction["prob"]
        
        # Determine if it's Healthy (Segar) or Rotten (Tidak Segar)
        is_healthy = "Healthy" in label
        category = "Segar (Fresh)" if is_healthy else "Tidak Segar (Not Fresh)"
        
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {prob:.2%}")
        st.write(f"**Category:** {category}")
        
        if st.button("✅ Confirm & Save", use_container_width=True):
            timestamp = datetime.now().isoformat()
            
            if label not in st.session_state.scan_data:
                st.session_state.scan_data[label] = []
            
            st.session_state.scan_data[label].append(timestamp)
            save_scan_data(st.session_state.scan_data)
            
            st.success("Data saved!")
            st.session_state.scan_triggered = False
            st.session_state.prediction = None
    else:
        st.info("👈 Go to Scanner tab, upload an image, and click 'Scan Image'")
    
    # Display Results dynamically
    st.markdown("---")
    st.subheader("📊 Current Scan Totals")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.write("**Segar (Fresh)**")
        
        healthy_items = {k: len(v) for k, v in st.session_state.scan_data.items() if "Healthy" in k}
        if healthy_items:
            for item, count in sorted(healthy_items.items()):
                st.write(f"{item}: **x{count}**")
        else:
            st.write("No fresh items yet.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.write("**Tidak Segar (Not Fresh)**")
        
        rotten_items = {k: len(v) for k, v in st.session_state.scan_data.items() if "Rotten" in k}
        if rotten_items:
            for item, count in sorted(rotten_items.items()):
                st.write(f"{item}: **x{count}**")
        else:
            st.write("No not-fresh items yet.")
        
        st.markdown("</div>", unsafe_allow_html=True)

# ========== TAB 3: DATA ==========
with tab3:
    st.markdown("<div class='tab-label'>📊 Data</div>", unsafe_allow_html=True)
    
    st.session_state.scan_data = load_scan_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Segar (Fresh)")
        healthy_items = {k: len(v) for k, v in st.session_state.scan_data.items() if "Healthy" in k}
        if healthy_items:
            for item, count in sorted(healthy_items.items()):
                st.write(f"{item}: **x{count}**")
        else:
            st.write("No fresh items.")
    
    with col2:
        st.subheader("Tidak Segar (Not Fresh)")
        rotten_items = {k: len(v) for k, v in st.session_state.scan_data.items() if "Rotten" in k}
        if rotten_items:
            for item, count in sorted(rotten_items.items()):
                st.write(f"{item}: **x{count}**")
        else:
            st.write("No not-fresh items.")
    
    st.markdown("---")
    if st.button("🗑️ Clear All Data", use_container_width=True):
        st.session_state.scan_data = {}
        save_scan_data(st.session_state.scan_data)
        st.success("All data cleared!")
        st.rerun()
