import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.title("تحليل ميول النصوص (Sentiment Analysis)")

MAX_LEN = 100  # طول التسلسل للنموذج، عدّل حسب تدريبك

@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model('sentiment_model.h5')
    with open('tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    return padded

def predict_sentiment(text):
    data = preprocess_text(text)
    pred = model.predict(data)
    class_idx = pred.argmax(axis=1)[0]
    classes = ['negative', 'neutral', 'positive']  # عدّل حسب تصنيفاتك
    return classes[class_idx]

user_input = st.text_area("اكتب نص لتحليل الميول:")

if st.button("تحليل"):
    if not user_input.strip():
        st.warning("من فضلك أدخل نصاً للتحليل.")
    else:
        result = predict_sentiment(user_input)
        st.success(f"الميول المتوقعة: {result}")

def predict_sentiment(text):
    data = preprocess_text(text)
    pred = model.predict(data)
    st.write("Raw model output:", pred)  # تظهر داخل التطبيق
    class_idx = pred.argmax(axis=1)[0]
    classes = ['negative', 'neutral', 'positive']
    return classes[class_idx]

