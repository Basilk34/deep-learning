import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from langdetect import detect
from geotext import GeoText

# تحميل النموذج المحفوظ بصيغة .h5
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('sentiment_model.h5')

# تحميل الـ tokenizer
@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        return pickle.load(handle)

# دالة التنبؤ
def preprocess_text(text, tokenizer, max_len=100):
    seq = tokenizer.texts_to_sequences([text])
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    return padded

def predict_sentiment(text, model, tokenizer):
    data = preprocess_text(text, tokenizer)
    pred = model.predict(data)
    class_idx = np.argmax(pred, axis=1)[0]
    classes = ['Extremely Negative', 'Negative', 'Neutral', 'Positive', 'Extremely Positive']
    confidence = pred[0][class_idx]
    return classes[class_idx], confidence

# كشف اللغة
def detect_language(text):
    try:
        return detect(text)
    except:
        return "Unknown"

# استخراج المواقع من النص
def extract_location(text):
    places = GeoText(text)
    return places.cities + places.countries

# واجهة التطبيق
st.title("تحليل ميول النصوص مع كشف اللغة والموقع")

model = load_model()
tokenizer = load_tokenizer()

user_input = st.text_area("📝 أدخل نصاً لتحليل ميوله:")

if st.button("تحليل"):
    if user_input.strip() == "":
        st.warning("الرجاء إدخال نص للتحليل!")
    else:
        # تصنيف التوجه
        sentiment, confidence = predict_sentiment(user_input, model, tokenizer)
        st.markdown(f"### 📊 ميول النص: **{sentiment}**")
        st.progress(confidence)

        # كشف اللغة
        language = detect_language(user_input)
        st.markdown(f"### 🌐 اللغة المتوقعة: **{language}**")

        # استخراج الموقع
        locations = extract_location(user_input)
        if locations:
            st.markdown(f"### 🌍 المواقع المذكورة في النص: **{', '.join(locations)}**")
        else:
            st.markdown("### ❌ لا يوجد موقع جغرافي واضح في النص")

