import os
os.system("python download_models.py")
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# تحميل الموديل العام للصور
general_model = tf.keras.models.load_model("models/general_model.h5")

# إعداد أسماء الكلاسات بالترتيب الصحيح
class_names = ['neutral', 'negative', 'positive']

# دالة لتحويل الصورة إلى شكل مناسب للموديل
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))  # نعدّل الحجم حسب ما تدرب عليه الموديل
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# واجهة Streamlit
st.title("📷 تحليل مشاعر الصور")
uploaded_image = st.file_uploader("ارفع صورة", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="الصورة المرفوعة", use_column_width=True)

    # تجهيز الصورة وتوقع النتيجة
    img_array = preprocess_image(image)
    preds = general_model.predict(img_array)
    predicted_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    # عرض النتيجة
    st.subheader("🔍 النتيجة:")
    st.write(f"التصنيف: **{predicted_class}**")
    st.write(f"الثقة: {confidence:.2f}%")
