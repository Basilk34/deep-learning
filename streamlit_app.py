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
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown
from PIL import Image

# حجم الصورة المطلوبة
IMG_SIZE = 224

# رابط التحميل المباشر للنموذج (استخرجت الـ FILE_ID من الرابط حقك)
MODEL_URL = 'https://drive.google.com/uc?export=download&id=1oYuyEpzubzQ2Ph67l3ZKvDjQYCcLwQjY'
MODEL_PATH = 'flower_model.h5'

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        st.write("⏳ جاري تحميل النموذج من Google Drive ...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    model = load_model(MODEL_PATH)
    return model

model = download_and_load_model()

def predict(img, model):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction

st.title("تصنيف صور الورود")

uploaded_file = st.file_uploader("ارفع صورة ورد هنا", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="الصورة التي رفعتها", use_column_width=True)
    
    prediction = predict(img, model)
    
    # ضع أسماء الأصناف حسب التدريب عندك
    class_labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    st.write(f"**التنبؤ:** {predicted_class}")
    st.write(f"**الثقة:** {confidence:.2f}%")

