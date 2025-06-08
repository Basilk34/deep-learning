import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import gdown
import os
import cv2

# أسماء الكلاسات
class_names = ['Neutral', 'negative', 'positive']

# روابط Google Drive
models_to_download = {
    "models/image_model_v1.h5": "18e9JcIpWWkRke1Rh2fEu_B1u6pbpzBh0",  # الصور العامة
    "models/image_model_v2.h5": "1QiS1oEYxnIbj3ykZ-OmfqmUj7u0yHYN3"   # الوجوه
}

# تأكد من تحميل الموديلات
os.makedirs("models", exist_ok=True)
for filepath, file_id in models_to_download.items():
    if not os.path.exists(filepath):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filepath, quiet=False)

# تحميل الموديلات
model_general = tf.keras.models.load_model("models/image_model_v1.h5")
model_faces = tf.keras.models.load_model("models/image_model_v2.h5")

# واجهة المستخدم
st.title("🧠 تحليل المشاعر من الصور (مع أو بدون وجه)")

uploaded_file = st.file_uploader("📷 ارفع صورة لتحليل المشاعر", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 الصورة المدخلة", use_column_width=True)

    # تحويل للصيغة التي يفهمها OpenCV
    original_img = np.array(image)
    gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

    # كشف الوجه
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_img = original_img[y:y+h, x:x+w]
        resized = cv2.resize(face_img, (224, 224))
        input_data = np.expand_dims(resized, axis=0)
        input_data = preprocess_input(input_data)
        prediction = model_faces.predict(input_data)
        model_used = "🤖 موديل الوجوه"
    else:
        resized = cv2.resize(original_img, (224, 224))
        input_data = np.expand_dims(resized, axis=0)
        input_data = preprocess_input(input_data)
        prediction = model_general.predict(input_data)
        model_used = "🖼️ موديل الصور العامة"

    predicted_class = class_names[np.argmax(prediction)]

    st.markdown(f"### ✅ التصنيف: `{predicted_class}`")
    st.markdown(f"### 📌 تم استخدام: `{model_used}`")

    st.markdown("### 🔢 احتمالات التصنيفات:")
    for cls, prob in zip(class_names, prediction[0]):
        st.write(f"**{cls}**: {prob:.2f}")
