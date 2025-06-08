import streamlit as st
import pickle
import numpy as np
import gdown
import os
import re
from collections import Counter
from googleapiclient.discovery import build
from youtube_comment_downloader import YoutubeCommentDownloader
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==================== إعداد ====================
MODEL_PATH = "arabic_sentiment_model.h5"  # موجود مسبقًا على GitHub
TOKENIZER_URL = "https://drive.google.com/uc?id=10jfNR3NcOh1MO2xpybPP9LrlRsTsHUzM"
TOKENIZER_PATH = "tokenizer.pkl"
YOUTUBE_API_KEY = "AIzaSyANEG0NbdmV_veIiZHY9cyK-0du_cYmtRk"
REGION = "SA"
MAX_COMMENTS = 50

# ============== تحميل التوكين من Drive ==============
@st.cache_resource
def download_tokenizer():
    if not os.path.exists(TOKENIZER_PATH):
        gdown.download(TOKENIZER_URL, TOKENIZER_PATH, quiet=False)
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)

# ============== وظائف المودل ==============
def is_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

def fetch_comments(video_id):
    downloader = YoutubeCommentDownloader()
    comments = []
    for c in downloader.get_comments_from_url(f"https://www.youtube.com/watch?v={video_id}", sort_by=1):
        if is_arabic(c["text"]):
            comments.append(c["text"])
        if len(comments) >= MAX_COMMENTS:
            break
    return comments

def predict(comments, tokenizer, model):
    sequences = tokenizer.texts_to_sequences(comments)
    padded = pad_sequences(sequences, maxlen=100)
    preds = model.predict(padded)
    labels = ['negative', 'neutral', 'positive']
    return list(zip(comments, [labels[np.argmax(p)] for p in preds]))

def summarize(results):
    pred = [s for _, s in results]
    counter = Counter(pred)
    total = len(pred)
    return {k: (v, (v / total) * 100) for k, v in counter.items()}

def get_trending_videos(query):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    req = youtube.search().list(q=query, part="snippet", type="video", regionCode=REGION, maxResults=10)
    res = req.execute()
    return [(item["snippet"]["title"], item["id"]["videoId"]) for item in res["items"]]

# ==================== Streamlit UI ====================
st.set_page_config(page_title="تحليل مشاعر يوتيوب", layout="centered")
st.title("🎥 تحليل مشاعر تعليقات YouTube")

query = st.text_input("🎯 اكتب موضوع للبحث (مثال: رمضان، كرة القدم، BTS):")

if query:
    videos = get_trending_videos(query)
    if videos:
        selected = st.selectbox("اختر فيديو:", [title for title, _ in videos])
        selected_id = dict(videos)[selected]
        st.video(f"https://www.youtube.com/watch?v={selected_id}")

        if st.button("🔍 تحليل التعليقات"):
            with st.spinner("جاري تحليل المشاعر..."):
                tokenizer = download_tokenizer()
                model = load_model(MODEL_PATH)
                comments = fetch_comments(selected_id)
                results = predict(comments, tokenizer, model)
                summary = summarize(results)

                st.subheader("📊 ملخص:")
                for label, (count, percent) in summary.items():
                    st.write(f"{label.capitalize()}: {count} تعليق ({percent:.2f}%)")

                st.subheader("🗣️ تعليقات مصنفة:")
                for comment, sentiment in results:
                    st.markdown(f"**{sentiment.upper()}** — {comment}")
    else:
        st.warning("❗ لم يتم العثور على فيديوهات.")
