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
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

# تحميل الموديل المدرب
model = tf.keras.models.load_model("models/image_model_v1.h5")

# أسماء الكلاسات
class_names = ['Neutral', 'negative', 'positive']

# عنوان الصفحة
st.title("🧠 تحليل المشاعر من الصور (بدون كشف وجه)")

# رفع صورة
uploaded_file = st.file_uploader("📷 ارفع صورة لتحليل المشاعر", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة الأصلية
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 الصورة المدخلة", use_column_width=True)

    # تجهيز الصورة للموديل
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # التنبؤ
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    # عرض النتيجة
    st.markdown(f"### ✅ التصنيف: `{predicted_class}`")
    
    # عرض الاحتمالات
    st.markdown("### 🔢 احتمالات التصنيفات:")
    for cls, prob in zip(class_names, prediction[0]):
        st.write(f"**{cls}**: {prob:.2f}")

