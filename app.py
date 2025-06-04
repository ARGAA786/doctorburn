import streamlit as st
import numpy as np
import cv2
import os
import gdown
import zipfile
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.preprocessing.image import img_to_array

# ------------------ تحميل الموديلات من Google Drive ------------------
def download_models():
    # تحميل segment.pt
    if not os.path.exists("segment.pt"):
        gdown.download("https://drive.google.com/uc?id=1Z5ndB6yh2go1ehKmioSAFD-1SxJ-kp9p", "segment.pt", quiet=False)

    # تحميل burn_classifier_saved_model.zip وفك الضغط
    if not os.path.exists("burn_classifier_saved_model"):
        gdown.download("https://drive.google.com/uc?id=1_MvC8rsTTTPz2oiQEd8MXCwiaIEUxUGF", "model.zip", quiet=False)
        with zipfile.ZipFile("model.zip", "r") as zip_ref:
            zip_ref.extractall("burn_classifier_saved_model")

# ------------------ تحميل النماذج ------------------
@st.cache_resource
def load_models():
    segment_model = YOLO("segment.pt")
    classifier_model = tf.saved_model.load("burn_classifier_saved_model")
    return segment_model, classifier_model

# ------------------ كلاس لابيلز ------------------
class_labels = ['First-degree burn', 'Second-degree burn', 'Third-degree burn']

# ------------------ إعداد صفحة Streamlit ------------------
st.set_page_config(page_title="Doctor Burn - Full System", layout="centered")
st.title("🔥 Doctor Burn")
st.write("Upload an image to detect and classify burn severity. (Kaggle-style segmentation + classification)")

# ------------------ تحميل الملفات ------------------
download_models()
segment_model, classifier_model = load_models()

# ------------------ رفع الصورة ------------------
uploaded_file = st.file_uploader("📤 Upload image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="🖼️ Original Image", use_column_width=True)

    # ------------------ Resize ل YOLO ------------------
    resized_input = cv2.resize(img_rgb.copy(), (640, 640))
    results = segment_model(resized_input)

    # ------------------ عرض النتيجة ------------------
    seg_img = results[0].plot()
    st.image(seg_img, caption="🧠 Segmented Burn Area", use_column_width=True)

    # ------------------ التصنيف ------------------
    resized_class_input = cv2.resize(img_rgb.copy(), (180, 180))
    arr = img_to_array(resized_class_input) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prediction = classifier_model(arr)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"🩺 **Prediction**: {class_labels[class_index]} ({confidence*100:.2f}%)")
