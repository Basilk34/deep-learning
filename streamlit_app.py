import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import gdown
import os

# أسماء الكلاسات
class_names = ['Neutral', 'negative', 'positive']

# روابط Google Drive بعد التعديل
models_to_download = {
    "models/image_model_faces.h5": "18e9JcIpWWkRke1Rh2fEu_B1u6pbpzBh0",  # 👤 موديل الوجوه
    "models/image_model_general.h5": "1QiS1oEYxnIbj3ykZ-OmfqmUj7u0yHYN3"   # 🖼️ موديل الصور العامة
}

# تحميل الموديلات
os.makedirs("models", exist_ok=True)
for filepath, file_id in models_to_download.items():
    if not os.path.exists(filepath):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filepath, quiet=False)

# تحميل الموديلات
model_faces = tf.keras.models.load_model("models/image_model_faces.h5")
model_general = tf.keras.models.load_model("models/image_model_general.h5")

# واجهة Streamlit
st.title("🧠 تحليل المشاعر من الصور")

# اختيار الموديل يدويًا
model_option = st.selectbox("اختر نوع الموديل:", ["👤 الوجوه فقط", "🖼️ الصور العامة"])

# رفع صورة
uploaded_file = st.file_uploader("📷 ارفع صورة", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 الصورة المدخلة", use_column_width=True)

    img_resized = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img_resized), axis=0)
    img_array = preprocess_input(img_array)

    # اختيار الموديل حسب المستخدم
    if model_option == "👤 الوجوه فقط":
        prediction = model_faces.predict(img_array)
        model_used = "👤 موديل الوجوه"
    else:
        prediction = model_general.predict(img_array)
        model_used = "🖼️ موديل الصور العامة"

    predicted_class = class_names[np.argmax(prediction)]

    st.markdown(f"### ✅ التصنيف: `{predicted_class}`")
    st.markdown(f"**🧠 الموديل المستخدم: `{model_used}`**")

    st.markdown("### 🔢 احتمالات التصنيفات:")
    for cls, prob in zip(class_names, prediction[0]):
        st.write(f"**{cls}**: {prob:.2f}")
