import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import gdown
import os

# إعداد أسماء الكلاسات
class_names = ['Neutral', 'negative', 'positive']

# إعداد روابط Google Drive
models_to_download = {
    "models/model_general.h5": "18e9JcIpWWkRke1Rh2fEu_B1u6pbpzBh0",  # الصور العامة
    "models/model_faces.h5": "1QiS1oEYxnIbj3ykZ-OmfqmUj7u0yHYN3"     # الوجوه
}

# إنشاء مجلد للموديلات
os.makedirs("models", exist_ok=True)

# تحميل الموديلات
for path, file_id in models_to_download.items():
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False)

# تحميل النماذج
model_general = tf.keras.models.load_model("models/model_general.h5")
model_faces = tf.keras.models.load_model("models/model_faces.h5")

# واجهة المستخدم
st.title("🧠 تحليل مشاعر من الصور")

# اختيار نوع الموديل
model_type = st.selectbox("👁️ اختر نوع التحليل:", ["تحليل الوجه", "تحليل الصورة الكاملة"])

# رفع الصورة
uploaded_file = st.file_uploader("📷 ارفع صورة", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 الصورة الأصلية", use_column_width=True)

    # معالجة الصورة
    resized = image.resize((224, 224))
    img_array = np.array(resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # اختيار الموديل المناسب
    if model_type == "تحليل الوجه":
        prediction = model_faces.predict(img_array)
        model_used = "👤 موديل الوجوه"
    else:
        prediction = model_general.predict(img_array)
        model_used = "🖼️ موديل الصور العامة"

    # النتيجة
    predicted_class = class_names[np.argmax(prediction)]

    st.markdown(f"### ✅ التصنيف: `{predicted_class}`")
    st.markdown(f"**🧠 النموذج المستخدم: `{model_used}`**")

    st.markdown("### 🔢 الاحتمالات:")
    for cls, prob in zip(class_names, prediction[0]):
        st.write(f"**{cls}**: {prob:.2f}")
