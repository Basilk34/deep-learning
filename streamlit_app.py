import streamlit as st
st.set_page_config(page_title="تحليل الميول", page_icon="💬")

import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
import re
import datetime
import matplotlib.pyplot as plt

# --- إعدادات ---
max_len = 100  # تأكد أنها نفس القيمة التي دربت بها
labels = ['Extremely Negative', 'Negative', 'Neutral', 'Positive', 'Extremely Positive']

# --- تحميل النموذج والتوكنيزر ---
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model('lstm_corona_model.h5')
    with open('tokenizer (1).pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# --- دالة استخراج الكلمات المفتاحية ---
def extract_keywords(text, num_keywords=5):
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = set(['the', 'is', 'in', 'and', 'to', 'of', 'a', 'for', 'on', 'with', 'that', 'this'])
    filtered_words = [w for w in words if w not in stopwords]
    most_common = Counter(filtered_words).most_common(num_keywords)
    return most_common  # ترجع قائمة من tuples (كلمة، تكرار)

# --- واجهة التطبيق ---
st.title("🔍 تحليل الميول تجاه كورونا باستخدام LSTM")

# أولاً: اسم المستخدم والتاريخ
username = st.text_input("👤 أدخل اسم المستخدم:")
date = st.date_input("📅 اختر التاريخ:", datetime.date.today())

# فقط إذا تم إدخال اسم المستخدم نعرض خانة النص وزر التحليل
if username:
    user_input = st.text_area("📝 أدخل تغريدة أو نص تحليل:")
    if st.button("تحليل"):
        if not user_input.strip():
            st.warning("الرجاء إدخال نص للتحليل!")
        else:
            # التنبؤ
            seq = tokenizer.texts_to_sequences([user_input])
            padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
            prediction = model.predict(padded)
            class_idx = np.argmax(prediction)
            confidence = np.max(prediction)

            # عرض النتائج
            st.success(f"**اسم المستخدم:** {username}")
            st.success(f"**التاريخ:** {date}")
            st.success(f"**التصنيف:** {labels[class_idx]}")
            st.info(f"**نسبة الثقة:** {confidence:.2f}")

            # استخراج الكلمات المفتاحية
            keywords = extract_keywords(user_input)
            if keywords:
                st.markdown("### 🔑 الكلمات المفتاحية الأكثر ظهورًا:")
                words, counts = zip(*keywords)
                st.write(", ".join(words))

                # رسم بياني
                fig, ax = plt.subplots()
                ax.bar(words, counts, color='skyblue')
                ax.set_ylabel('عدد التكرار')
                ax.set_title('الكلمات المفتاحية')
                st.pyplot(fig)
else:
    st.info("الرجاء إدخال اسم المستخدم للبدء.")

import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import gdown
import os
from PIL import Image

# إعداد المتغيرات
IMG_SIZE = 224
MODEL_PATH = "flower_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1oYuyEpzubzQ2Ph67l3ZKvDjQYCcLwQjY"

# تحميل النموذج من Google Drive لو مش موجود محلياً
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# تحميل النموذج
model = tf.keras.models.load_model(MODEL_PATH)

# دالة التنبؤ
def predict_flower(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_labels = list(train_generator.class_indices.keys()) if 'train_generator' in globals() else ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

# عنوان التطبيق
st.title("Flower Classification with VGG16")

# رفع صورة من المستخدم
uploaded_file = st.file_uploader("Upload an image of a flower", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    predicted_class, confidence = predict_flower(img)
    st.write(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: {confidence:.2f}")

else:
    st.write("Please upload an image to classify.")
