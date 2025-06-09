import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from youtube_sentiment import analyze_youtube_topic

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

def analyze_youtube(topic):
    try:
        analysis = analyze_youtube_topic(topic)
        title = analysis["video_title"]
        url = analysis["video_url"]
        results = analysis["results"]
        summary = analysis["summary"]

        result_text = f"📺 **{title}**\n🔗 {url}\n\n"

        for comment, sentiment in results[:10]:
            result_text += f"🗣️ {comment}\n➡️ {sentiment}\n\n"

        summary_text = "📊 **Summary**:\n"
        for k, v in summary.items():
            summary_text += f"{k}: {v} تعليق\n"

        return result_text, summary_text

    except Exception as e:
        return "❌ حدث خطأ: " + str(e), ""

# Gradio interface
with gr.Blocks(title="مشروع هنودي لتحليل المشاعر 👑") as demo:
    gr.Markdown("# 👑 مشروع تحليل مشاعر الصور والنصوص والتعليقات")

    with gr.Tab("📷 تحليل الصور"):
        image_input = gr.Image(type="numpy", label="ارفع صورة")
        image_output = gr.Label(label="تحليل الصورة")
        image_input.change(fn=classify_image, inputs=image_input, outputs=image_output)

    with gr.Tab("📝 تحليل النصوص"):
        text_input = gr.Textbox(label="اكتب نصًا")
        text_output = gr.Label(label="تحليل النص")
        text_input.change(fn=classify_text, inputs=text_input, outputs=text_output)

    with gr.Tab("📹 تحليل ترند اليوتيوب"):
        topic_input = gr.Textbox(label="ادخل موضوع مثل 'كرة القدم' أو 'أغاني'")
        youtube_results = gr.Markdown()
        youtube_summary = gr.Markdown()
        analyze_btn = gr.Button("🔍 تحليل تعليقات ترند")
        analyze_btn.click(fn=analyze_youtube, inputs=topic_input, outputs=[youtube_results, youtube_summary])

demo.launch()
