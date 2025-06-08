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

