import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical

# 1. بيانات بسيطة
texts = [
    "I love this movie",
    "This is terrible",
    "It's okay, not bad",
    "Amazing work!",
    "I hate it",
    "Nothing special"
]
labels = [2, 0, 1, 2, 0, 1]  # 2=إيجابي، 0=سلبي، 1=محايد
labels_cat = to_categorical(labels, num_classes=3)

# 2. تجهيز النصوص
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_seqs = pad_sequences(sequences, maxlen=10)

# 3. بناء نموذج بسيط
model = Sequential([
    Embedding(1000, 32, input_length=10),
    LSTM(32),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(padded_seqs, labels_cat, epochs=50, verbose=0)

# 4. واجهة المستخدم
st.title("🔍 تحليل التوجه من النصوص")
st.write("أدخل جملة وسيقوم النموذج بتحليل التوجه الخاص بها (سلبي، محايد، إيجابي).")

user_input = st.text_input("✍️ أدخل النص هنا:")

if user_input:
    seq = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(seq, maxlen=10)
    pred = model.predict(padded)
    label = np.argmax(pred)
    label_map = ["سلبي", "محايد", "إيجابي"]
    st.success(f"التوجه المتوقع: **{label_map[label]}**")
