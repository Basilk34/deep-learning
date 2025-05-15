import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# 1. تحميل النموذج
model = tf.keras.models.load_model('sentiment_model')

# 2. تحميل Tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_len = 100  # نفس قيمة max_len أثناء التدريب

# دالة لتحضير النصوص
def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    return padded

# دالة التوقع
def predict_sentiment(text):
    data = preprocess_text(text)
    pred = model.predict(data)
    class_idx = pred.argmax(axis=1)[0]
    classes = ['negative', 'neutral', 'positive']  # عدل حسب بياناتك
    return classes[class_idx]

# واجهة Streamlit
st.title("تحليل ميول الناس من تويتر")

user_input = st.text_input("اكتب النص هنا لتحليل المشاعر:")

if st.button("تحليل"):
    if user_input.strip() == "":
        st.warning("من فضلك أدخل نص لتحليله")
    else:
        result = predict_sentiment(user_input)
        st.success(f"الميول: {result}")
