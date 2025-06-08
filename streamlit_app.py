import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import mediapipe as mp
from tensorflow.keras.applications.vgg16 import preprocess_input
import gdown
import os

# أسماء التصنيفات
class_names = ['Neutral', 'negative', 'positive']

# روابط الموديلات
models_to_download = {
    "models/model_faces.h5": "18e9JcIpWWkRke1Rh2fEu_B1u6pbpzBh0",     # موديل الوجوه
    "models/model_general.h5": "1QiS1oEYxnIbj3ykZ-OmfqmUj7u0yHYN3"    # موديل الصور العامة
}

# تحميل الموديلات من Google Drive
os.makedirs("models", exist_ok=True)
for path, file_id in models_to_download.items():
    if not os.path.exists(path):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)

# تحميل النماذج
model_faces = tf.keras.models.load_model("models/model_faces.h5")
model_general = tf.keras.models.load_model("models/model_general.h5")

# إعداد mediapipe
mp_face_detection = mp.solutions.face_detection

# واجهة المستخدم
st.title("🧠 تحليل المشاعر من الصور")

uploaded_file = st.file_uploader("📷 ارفع صورة", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image, caption="📷 الصورة الأصلية", use_column_width=True)

    face_detected = False
    face_crop = None

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
        results = detector.process(image_np)
        if results.detections:
            # تم الكشف عن وجه
            face_detected = True
            bbox = results.detections[0].location_data.relative_bounding_box
            ih, iw, _ = image_np.shape
            x = int(bbox.xmin * iw)
            y = int(bbox.ymin * ih)
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)
            face_crop = image_np[y:y+h, x:x+w]

    if face_detected:
        st.info("✅ تم الكشف عن وجه، سيتم استخدام موديل الوجوه")
        face_resized = Image.fromarray(face_crop).resize((224, 224))
        img_array = np.expand_dims(np.array(face_resized), axis=0)
        img_array = preprocess_input(img_array)
        prediction = model_faces.predict(img_array)
        st.image(face_crop, caption="🧑‍🦱 الوجه المقطوع", use_column_width=False)
        model_used = "👤 موديل الوجوه"
    else:
        st.warning("⚠️ لم يتم الكشف عن وجه، سيتم استخدام موديل الصور العامة")
        full_resized = image.resize((224, 224))
        img_array = np.expand_dims(np.array(full_resized), axis=0)
        img_array = preprocess_input(img_array)
        prediction = model_general.predict(img_array)
        model_used = "🖼️ موديل الصورة الكاملة"

    predicted_class = class_names[np.argmax(prediction)]
    st.success(f"✅ التصنيف النهائي: `{predicted_class}`")
    st.markdown(f"🧠 النموذج المستخدم: **{model_used}**")

    st.markdown("### 🔢 احتمالات التصنيفات:")
    for cls, prob in zip(class_names, prediction[0]):
        st.write(f"**{cls}**: {prob:.2f}")
