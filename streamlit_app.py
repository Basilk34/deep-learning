import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

# --- واجهة Streamlit ---
st.set_page_config(page_title="تحليل الميول", page_icon="💬")
st.title("🔍 تحليل الميول تجاه كورونا باستخدام LSTM")

user_input = st.text_input("📝 أدخل تغريدة أو نص تحليل:")

if user_input:
    # تحويل النص إلى تسلسل رقمي
    seq = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

    # التنبؤ
    prediction = model.predict(padded)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)

    # عرض النتيجة
    st.success(f"**التصنيف:** {labels[class_idx]}")
    st.info(f"**نسبة الثقة:** {confidence:.2f}")
