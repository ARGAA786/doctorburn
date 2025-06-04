import streamlit as st
import numpy as np
import cv2
import gdown
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from ultralytics import YOLO

# تحميل الموديلات من Google Drive (لو مش موجودين)
SEGMENT_MODEL_ID = "1Z5ndB6yh2go1ehKmioSAFD-1SxJ-kp9p"
CLASSIFIER_MODEL_ID = "1_MvC8rsTTTPz2oiQEd8MXCwiaIEUxUGF"

def download_file_from_drive(file_id, output):
    if not os.path.exists(output):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

download_file_from_drive(SEGMENT_MODEL_ID, "segment.pt")
download_file_from_drive(CLASSIFIER_MODEL_ID, "burn_classifier_saved_model.zip")

# فك الضغط للموديل المصنف
import zipfile
if not os.path.exists("burn_classifier_saved_model"):
    with zipfile.ZipFile("burn_classifier_saved_model.zip", 'r') as zip_ref:
        zip_ref.extractall("burn_classifier_saved_model")

# تحميل الموديلات
@st.cache_resource
def load_models():
    segment_model = YOLO("segment.pt")
    classifier_model = tf.saved_model.load("burn_classifier_saved_model")
    return segment_model, classifier_model

segment_model, classifier_model = load_models()

class_labels = ['First-degree burn', 'Second-degree burn', 'Third-degree burn']

# واجهة Streamlit
st.set_page_config(page_title="Doctor Burn", layout="centered")
st.title("🔥 Doctor Burn")
st.write("Upload an image to detect and classify burn severity.")

uploaded_file = st.file_uploader("📁 Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="🖼️ Original Image", use_column_width=True)

    resized_input = cv2.resize(img_rgb.copy(), (640, 640))
    results = segment_model(resized_input)
    result_img = results[0].plot()
    st.image(result_img, caption="🩹 Segmentation", use_column_width=True)

    # تصنيف نوع الحرق
    resized_for_classifier = cv2.resize(img_rgb, (224, 224))
    x = img_to_array(resized_for_classifier)
    x = np.expand_dims(x, axis=0)
    prediction = classifier_model(x)
    class_index = np.argmax(prediction)
    st.success(f"🧠 Prediction: {class_labels[class_index]}")
