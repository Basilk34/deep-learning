import streamlit as st
import re
import numpy as np
import pickle
import gdown
from collections import Counter
from googleapiclient.discovery import build
from youtube_comment_downloader import YoutubeCommentDownloader
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# ============== CONFIG ==============
YOUTUBE_API_KEY = "AIzaSyANEG0NbdmV_veIiZHY9cyK-0du_cYmtRk"
MODEL_URL = "https://drive.google.com/uc?id=1yrWFfq7HDUt2kgpOaG0oUMrz1vjxjWQs"
TOKENIZER_URL = "https://drive.google.com/uc?id=10jfNR3NcOh1MO2xpybPP9LrlRsTsHUzM"
MODEL_PATH = "arabic_sentiment_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
REGION_CODE = "SA"
MAX_COMMENTS = 50

# ============== DOWNLOAD FILES ==============
@st.cache_resource
def download_files():
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    gdown.download(TOKENIZER_URL, TOKENIZER_PATH, quiet=False)

# ============== LOAD MODEL & TOKENIZER ==============
@st.cache_resource
def load_model_tokenizer():
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    model = tf.keras.models.load_model(MODEL_PATH)
    return tokenizer, model

# ============== CHECK ARABIC ==============
def is_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

# ============== FETCH VIDEO COMMENTS ==============
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

# ============== PREDICT SENTIMENT ==============
def predict_sentiment(comments, tokenizer, model, max_len=100):
    sequences = tokenizer.texts_to_sequences(comments)
    padded = pad_sequences(sequences, maxlen=max_len)
    preds = model.predict(padded)
    labels = ['negative', 'neutral', 'positive']
    return list(zip(comments, [labels[np.argmax(p)] for p in preds]))

def summarize_results(results):
    predictions = [s for _, s in results]
    counter = Counter(predictions)
    total = len(predictions)
    return {
        label: (count, (count / total) * 100 if total else 0)
        for label, count in counter.items()
    }

# ============== FETCH VIDEOS ==============
def search_trending_videos(api_key, query, region='SA', max_results=10):
    youtube = build('youtube', 'v3', developerKey=api_key)
    response = youtube.search().list(
        q=query,
        part='snippet',
        type='video',
        regionCode=region,
        maxResults=max_results
    ).execute()
    results = []
    for item in response.get("items", []):
        video_id = item["id"]["videoId"]
        title = item["snippet"]["title"]
        url = f"https://www.youtube.com/watch?v={video_id}"
        results.append((title, url, video_id))
    return results

# ============== UI ===================
st.set_page_config(page_title="تحليل مشاعر يوتيوب", layout="centered")
st.title("🎥 تحليل مشاعر تعليقات فيديوهات YouTube")
st.write("اكتب موضوع، اختر فيديو، وخلينا نشوف شو المشاعر 👇")

query = st.text_input("🎯 الموضوع:")

if query:
    with st.spinner("📡 جاري جلب الفيديوهات..."):
        videos = search_trending_videos(YOUTUBE_API_KEY, query, REGION_CODE, 10)
        if videos:
            options = {f"{title}": (video_id, url) for title, url, video_id in videos}
            selection = st.selectbox("اختر فيديو:", list(options.keys()))
            video_id, video_url = options[selection]
            st.video(video_url)

            if st.button("🔍 تحليل التعليقات"):
                with st.spinner("🧠 جاري تحليل المشاعر..."):
                    download_files()
                    tokenizer, model = load_model_tokenizer()
                    comments = fetch_arabic_comments(video_id, MAX_COMMENTS)
                    results = predict_sentiment(comments, tokenizer, model)
                    summary = summarize_results(results)

                    st.subheader("📊 نتائج المشاعر:")
                    for sentiment, (count, percent) in summary.items():
                        st.write(f"- {sentiment.capitalize()}: {count} تعليق ({percent:.2f}%)")

                    st.markdown("---")
                    st.subheader("🗣️ تعليقات مصنفة:")
                    for comment, sentiment in results:
                        st.markdown(f"**{sentiment.upper()}** — {comment}")
        else:
            st.warning("⚠️ لا يوجد فيديوهات مطابقة.")
