# 1. تحميل gdown وتنزيل الموديلات من Google Drive
import gdown
import os

os.makedirs("models", exist_ok=True)

# روابط Google Drive
image_model_v1_id = "18e9JcIpWWkRke1Rh2fEu_B1u6pbpzBh0"
image_model_v2_id = "1QiS1oEYxnIbj3ykZ-OmfqmUj7u0yHYN3"

gdown.download(f"https://drive.google.com/uc?id={image_model_v1_id}", "models/image_model_v1.h5", quiet=False)
gdown.download(f"https://drive.google.com/uc?id={image_model_v2_id}", "models/image_model_v2.h5", quiet=False)

# لو عندك موديل نص وتوكينايزر ضيفهم بنفس الطريقة
# gdown.download(...)

# 2. استيراد Streamlit والموديلات بعد ما نزلوا
import streamlit as st
import tensorflow as tf

# تحميل الموديلات
model_img1 = tf.keras.models.load_model("models/image_model_v1.h5")
model_img2 = tf.keras.models.load_model("models/image_model_v2.h5")

# ... كمل بعد هيك الواجهة والتصنيف
