import streamlit as st
import os
import pickle
import numpy as np
import re
from collections import Counter
from googleapiclient.discovery import build
from youtube_comment_downloader import YoutubeCommentDownloader
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# 🔐 CONFIG
# ============================
YOUTUBE_API_KEY = "AIzaSyANEG0NbdmV_veIiZHY9cyK-0du_cYmtRk"
TOKENIZER_PATH = "tokenizer.pkl"
MODEL_PATH = "arabic_sentiment_model.h5"
LABELS = ['negative', 'neutral', 'positive']
MAX_COMMENTS = 50

# ============================
# 📥 Load Tokenizer and Model
# ============================
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ============================
# 🔍 Helper Functions
# ============================
def is_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

def search_trending_videos_by_topic(api_key, topic, max_results=5):
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.search().list(
        part='snippet',
        q=topic,
        maxResults=max_results,
        type='video',
        relevanceLanguage='ar',
        regionCode='SA'
    )
    response = request.execute()
    return [{
        'title': item['snippet']['title'],
        'videoId': item['id']['videoId']
    } for item in response.get('items', [])]

def fetch_arabic_comments(video_id, max_comments=50):
    downloader = YoutubeCommentDownloader()
    raw_comments = []
    for comment in downloader.get_comments_from_url(f"https://www.youtube.com/watch?v={video_id}", sort_by=1):
        text = comment['text']
        if is_arabic(text):
            raw_comments.append(text)
        if len(raw_comments) >= max_comments:
            break
    return raw_comments

def predict_sentiment(comments, max_len=100):
    sequences = tokenizer.texts_to_sequences(comments)
    padded = pad_sequences(sequences, maxlen=max_len)
    preds = model.predict(padded)
    return list(zip(comments, [LABELS[np.argmax(p)] for p in preds]))

def summarize_results(results):
    predictions = [s for _, s in results]
    counter = Counter(predictions)
    total = len(predictions)
    return {
        label: (count, (count / total) * 100 if total else 0)
        for label, count in counter.items()
    }

# ============================
# 🎯 Streamlit App
# ============================
st.title("🎬 تحليل مشاعر تعليقات يوتيوب حسب موضوع معين")

topic = st.text_input("📝 أدخل موضوع تريد البحث عنه (مثال: كرة قدم، دراما، موسيقى):")

if topic:
    st.info("🔍 جاري البحث عن فيديوهات ترند...")
    videos = search_trending_videos_by_topic(YOUTUBE_API_KEY, topic, max_results=5)

    if videos:
        selected_video = st.selectbox("📺 اختر فيديو لتحليل التعليقات:", [v['title'] for v in videos])
        selected_id = next(v['videoId'] for v in videos if v['title'] == selected_video)
        video_url = f"https://www.youtube.com/watch?v={selected_id}"
        st.markdown(f"🔗 [رابط الفيديو]({video_url})")

        if st.button("🚀 تحليل التعليقات"):
            with st.spinner("📥 جاري تحميل وتحليل التعليقات..."):
                comments = fetch_arabic_comments(selected_id, MAX_COMMENTS)
                if not comments:
                    st.warning("❌ لم يتم العثور على تعليقات عربية.")
                else:
                    results = predict_sentiment(comments)
                    summary = summarize_results(results)

                    df = pd.DataFrame(results, columns=["التعليق", "التصنيف"])
                    st.markdown("## 📝 التعليقات المصنفة:")
                    st.dataframe(df)

                    st.markdown("## 📊 ملخص التصنيفات:")
                    for label, (count, percent) in summary.items():
                        st.write(f"**{label}**: {count} تعليق ({percent:.2f}%)")

                    st.markdown("## 📈 الرسم البياني:")
                    fig, ax = plt.subplots()
                    ax.bar(summary.keys(), [v[0] for v in summary.values()])
                    st.pyplot(fig)
    else:
        st.warning("❌ لم يتم العثور على فيديوهات ترند للموضوع.")
