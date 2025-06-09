# ✅ streamlit_app.py - تحليل مشاعر تعليقات يوتيوب بالعربي

import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from googleapiclient.discovery import build
from youtube_comment_downloader import YoutubeCommentDownloader
from collections import Counter
import re

# ========== الإعدادات العامة ==========
MODEL_PATH = "arabic_sentiment_model_clean.h5"
TOKENIZER_PATH = "tok.pkl"
YOUTUBE_API_KEY = "AIzaSyANEG0NbdmV_veIiZHY9cyK-0du_cYmtRk"
REGION = "SA"
MAX_COMMENTS = 50
MAX_LEN = 100
LABELS = ['negative', 'neutral', 'positive']

# ========== تحميل التوكينايزر والموديل ==========
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

model = load_model(MODEL_PATH)

# ========== الدوال ==========
def is_arabic(text):
    return bool(re.search(r"[\u0600-\u06FF]", text))

def get_trending_videos(api_key, query, region=REGION, max_results=10):
    youtube = build('youtube', 'v3', developerKey=api_key)
    response = youtube.search().list(
        q=query,
        part='snippet',
        type='video',
        regionCode=region,
        maxResults=max_results
    ).execute()
    return [(item['snippet']['title'], item['id']['videoId']) for item in response.get("items", [])]

def get_arabic_comments(video_id, max_comments=50):
    downloader = YoutubeCommentDownloader()
    raw_comments = []
    for comment in downloader.get_comments_from_url(f"https://www.youtube.com/watch?v={video_id}", sort_by=1):
        text = comment['text']
        if is_arabic(text):
            raw_comments.append(text)
        if len(raw_comments) >= max_comments:
            break
    return raw_comments

def predict_sentiments(comments):
    sequences = tokenizer.texts_to_sequences(comments)
    padded = pad_sequences(sequences, maxlen=MAX_LEN)
    predictions = model.predict(padded)
    labels = [LABELS[np.argmax(p)] for p in predictions]
    return list(zip(comments, labels))

def summarize(predictions):
    stats = Counter([label for _, label in predictions])
    total = sum(stats.values())
    return {label: f"{count} ({(count/total)*100:.1f}%)" for label, count in stats.items()}

# ========== واجهة Streamlit ==========
st.set_page_config(page_title="تحليل مشاعر ترند يوتيوب بالعربي")
st.title("🔍 مشروع تحليل المشاعر من تعليقات يوتيوب")
st.markdown("اكتب موضوع ترند، اختر فيديو، وراح يتم تحليل مشاعر التعليقات بالعربي.")

query = st.text_input("🎯 اكتب موضوع للبحث عن فيديوهات ترند:")

if query:
    videos = get_trending_videos(YOUTUBE_API_KEY, query)
    if not videos:
        st.warning("🚫 لا يوجد فيديوهات على هذا الموضوع")
    else:
        titles = [v[0] for v in videos]
        selected = st.selectbox("🎬 اختر فيديو: ", titles)
        video_id = dict(videos)[selected]

        if st.button("🔍 تحليل التعليقات"):
            with st.spinner("جاري تحليل التعليقات..."):
                comments = get_arabic_comments(video_id, MAX_COMMENTS)
                results = predict_sentiments(comments)
                stats = summarize(results)

                st.success("✅ تم التحليل")

                for comment, label in results:
                    st.markdown(f"**{label.upper()}** ➤ {comment}")

                st.markdown("---")
                st.subheader("📊 ملخص النتائج:")
                for label in LABELS:
                    st.write(f"{label}: {stats.get(label, '0 (0.0%)')}")
