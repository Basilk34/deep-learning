import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from PIL import Image
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('sentiment_model.h5')

st.title("تحليل ميول الناس من النصوص والصور")

# -- إعدادات نموذج النص --
max_len = 100

# تحميل نموذج النص والtokenizer
@st.cache(allow_output_mutation=True)
def load_text_model():
    model = tf.keras.models.load_model('sentiment_model_text')
    with open('tokenizer_text.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

text_model, tokenizer = load_text_model()

def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    return padded

def predict_text_sentiment(text):
    data = preprocess_text(text)
    pred = text_model.predict(data)
    class_idx = pred.argmax(axis=1)[0]
    classes = ['negative', 'neutral', 'positive']  # عدل حسب نموذجك
    return classes[class_idx]

# -- واجهة نص --
st.header("تحليل النص")
user_text = st.text_input("أدخل نص التحليل:")

if st.button("تحليل النص"):
    if user_text.strip() == "":
        st.warning("يرجى إدخال نص")
    else:
        result = predict_text_sentiment(user_text)
        st.success(f"الميول: {result}")

# -- إعدادات نموذج الصورة --
IMG_SIZE = (224, 224)  # حسب مدخلات نموذج الصورة

@st.cache(allow_output_mutation=True)
def load_image_model():
    model = tf.keras.models.load_model('sentiment_model_image')
    return model

image_model = load_image_model()

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image_sentiment(image):
    img = preprocess_image(image)
    pred = image_model.predict(img)
    class_idx = pred.argmax(axis=1)[0]
    classes = ['negative', 'neutral', 'positive']  # عدل حسب نموذجك
    return classes[class_idx]

# -- واجهة صورة --
st.header("تحليل الصورة")
uploaded_file = st.file_uploader("ارفع صورة", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="الصورة المرفوعة", use_column_width=True)
    if st.button("تحليل الصورة"):
        result_img = predict_image_sentiment(image)
        st.success(f"الميول من الصورة: {result_img}")
