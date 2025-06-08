import streamlit as st
import gdown
import os

# إنشاء مجلد الموديلات
os.makedirs("models", exist_ok=True)

# ملفات التحميل (اسم الملف → Google Drive File ID)
models_to_download = {
    "models/image_model_v1.h5": "18e9JcIpWWkRke1Rh2fEu_B1u6pbpzBh0",
    "models/image_model_v2.h5": "1QiS1oEYxnIbj3ykZ-OmfqmUj7u0yHYN3"
}

st.subheader("📥 تحميل الموديلات من Google Drive")

for filepath, file_id in models_to_download.items():
    if not os.path.exists(filepath):
        try:
            url = f"https://drive.google.com/uc?id={file_id}"
            st.info(f"🔄 جاري تحميل: {os.path.basename(filepath)}")
            gdown.download(url, filepath, quiet=False)
        except Exception as e:
            st.error(f"❌ مشكلة في تحميل {os.path.basename(filepath)}")
    else:
        st.success(f"✅ {os.path.basename(filepath)} تم تحميله مسبقاً")

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

# تحميل الموديلات (وجه + صورة عامة)
model_general = tf.keras.models.load_model("models/image_model_v1.h5")
model_faces = tf.keras.models.load_model("models/image_model_v2.h5")

# أسماء التصنيفات
class_names = ['Neutral', 'negative', 'positive']

# عنوان
st.title("🔍 تحليل مشاعر من الصور باستخدام موديل الوجه وموديل الصورة الكاملة")

# رفع صورة من المستخدم
uploaded_file = st.file_uploader("📷 ارفع صورة لتحليل المشاعر", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    original_img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

    # كاشف الوجوه
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # تحديد الطريقة
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_img = original_img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))
        x_input = np.expand_dims(face_img, axis=0)
        x_input = preprocess_input(x_input)
        prediction = model_faces.predict(x_input)
        model_used = "🧠 موديل الوجوه"
    else:
        full_img = cv2.resize(original_img, (224, 224))
        x_input = np.expand_dims(full_img, axis=0)
        x_input = preprocess_input(x_input)
        prediction = model_general.predict(x_input)
        model_used = "🧠 موديل الصورة الكاملة"

    # تصنيف النتيجة
    predicted_class = class_names[np.argmax(prediction)]

    # عرض الصورة والنتائج
    st.image(image, caption="📷 الصورة المدخلة", use_column_width=True)
    st.markdown(f"### 🧪 التصنيف: `{predicted_class}`")
    st.markdown(f"**✅ تم استخدام: `{model_used}`**")

    # الاحتمالات
    st.markdown("### 🔢 احتمالات التصنيفات:")
    for cls, prob in zip(class_names, prediction[0]):
        st.write(f"**{cls}**: {prob:.2f}")
