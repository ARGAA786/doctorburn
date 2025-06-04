import os
import zipfile
import gdown

# 1. Download segment.pt if not found
if not os.path.exists("segment.pt"):
    gdown.download("https://drive.google.com/uc?id=1Z5ndB6yh2go1ehKmioSAFD-1SxJ-kp9p", "segment.pt", quiet=False)

# 2. Download and extract burn_classifier_saved_model if not found
if not os.path.exists("burn_classifier_saved_model"):
    gdown.download("https://drive.google.com/uc?id=1_MvC8rsTTTPz2oiQEd8MXCwiaIEUxUGF", "burn_model.zip", quiet=False)
    with zipfile.ZipFile("burn_model.zip", 'r') as zip_ref:
        zip_ref.extractall("burn_classifier_saved_model")

import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import os
import warnings

st.set_page_config(page_title="Doctor Burn - Full System", layout="centered")
st.title("🔥 Doctor Burn")
st.write("Upload an image to detect and classify burn severity. (Kaggle-style segmentation + classification)")

# تحميل الموديلات
@st.cache_resource
def load_models():
    segment_model = YOLO("segment.pt")
    classifier_model = tf.saved_model.load("burn_classifier_saved_model")
    return segment_model, classifier_model

segment_model, classifier_model = load_models()
class_labels = ['First-degree burn', 'Second-degree burn', 'Third-degree burn']

# رفع الصورة
uploaded_file = st.file_uploader("📤 Upload image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="📷 Original Image", use_column_width=True)

    # Resize لـ YOLO
    resized_input = cv2.resize(img_rgb.copy(), (640, 640))
    results = segment_model(resized_input)

    if results[0].masks is not None:
        # خد أول ماسك زي كاجل
        mask_tensor = results[0].masks[0]
        mask_np = mask_tensor.data.cpu().numpy()

        if len(mask_np.shape) == 3:
            mask_np = mask_np[0]

        mask_np = (mask_np > 0.5).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask_np, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # الخلفية البيضا
        white_bg = np.ones_like(img_rgb) * 255
        result_img = white_bg.copy()
        result_img[mask_resized > 0] = img_rgb[mask_resized > 0]

        st.image(result_img, caption="✅ Segmented Burn Area", use_column_width=True)

        # استخراج الـ crop لمنطقة الحرق
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            cropped = result_img[y:y+h, x:x+w]

            resized = cv2.resize(cropped, (224, 224))
            array = img_to_array(resized) / 255.0
            array = np.expand_dims(array, axis=0)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                infer = classifier_model.signatures["serving_default"]
                result = infer(tf.convert_to_tensor(array, dtype=tf.float32))
                predicted_class = class_labels[np.argmax(list(result.values())[0].numpy())]

            st.success(f"🧠 **Burn Type Prediction**: {predicted_class}")
        else:
            st.warning("⚠️ Could not find valid burn region to classify.")
    else:
        st.error("🚫 No burn area detected in the image.")
