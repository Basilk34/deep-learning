import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load models
image_model = tf.keras.models.load_model("models/general_model.h5")
text_model = tf.keras.models.load_model("models/text_model.h5")
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

image_labels = ['neutral', 'negative', 'positive']
text_labels = ['negative', 'neutral', 'positive']

def classify_image(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    arr = np.expand_dims(img, axis=0)
    pred = image_model.predict(arr)[0]
    return dict(zip(image_labels, map(float, pred)))

def classify_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    pred = text_model.predict(padded)[0]
    return dict(zip(text_labels, map(float, pred)))

app = gr.Interface(
    fn=[classify_image, classify_text],
    inputs=[gr.Image(type="numpy", label="صورة"), gr.Textbox(label="نص عربي")],
    outputs=[gr.Label(label="تحليل الصورة"), gr.Label(label="تحليل النص")],
    title="📊 هنودي - مشروع تحليل مشاعر الصور والنصوص 👑"
)

app.launch()
