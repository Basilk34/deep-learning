import re
import pickle
import numpy as np
import random
from collections import Counter
from googleapiclient.discovery import build
from youtube_comment_downloader import YoutubeCommentDownloader
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

YOUTUBE_API_KEY = "AIzaSyANEG0NbdmV_veIiZHY9cyK-0du_cYmtRk"
MAX_COMMENTS = 50
SAMPLE_FROM_TOP = 5
REGION_CODE = "SA"

# Load the model and tokenizer
text_model = tf.keras.models.load_model("models/text_model.h5")
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

labels = ['negative', 'neutral', 'positive']

def is_arabic(text):
    return bool(re.search(r'[؀-ۿ]', text))

def search_trending_video_by_topic(api_key, topic, region='SA', sample_from=5):
    youtube = build('youtube', 'v3', developerKey=api_key)
    search_response = youtube.search().list(
        q=topic,
        part='snippet',
        regionCode=region,
        maxResults=sample_from,
        type='video'
    ).execute()

    if not search_response.get('items'):
        raise Exception("❌ لم يتم العثور على فيديوهات ترند للموضوع المحدد.")

    item = random.choice(search_response['items'])
    video_id = item['id']['videoId']
    title = item['snippet']['title']
    return video_id, title

def fetch_arabic_comments(video_id, max_comments=50):
    downloader = YoutubeCommentDownloader()
    raw_comments = []
    for comment in downloader.get_comments_from_url(f"https://www.youtube.com/watch?v={video_id}", sort_by=1):
        text = comment['text']
        if is_arabic(text):
            votes_raw = comment.get("votes", "0")
            votes = 0
            try:
                if "K" in votes_raw:
                    votes = int(float(votes_raw.replace("K", "")) * 1000)
                else:
                    votes = int(votes_raw)
            except:
                votes = 0
            raw_comments.append({"text": text, "votes": votes})
        if len(raw_comments) >= max_comments:
            break
    return [c["text"] for c in sorted(raw_comments, key=lambda x: x["votes"], reverse=True)]

def predict_sentiment(comments, max_len=100):
    sequences = tokenizer.texts_to_sequences(comments)
    padded = pad_sequences(sequences, maxlen=max_len)
    preds = text_model.predict(padded)
    return list(zip(comments, [labels[np.argmax(p)] for p in preds]))

def analyze_youtube_topic(topic):
    video_id, video_title = search_trending_video_by_topic(YOUTUBE_API_KEY, topic, REGION_CODE, SAMPLE_FROM_TOP)
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    comments = fetch_arabic_comments(video_id, MAX_COMMENTS)
    results = predict_sentiment(comments)

    summary = Counter([sent for _, sent in results])
    return {
        "video_title": video_title,
        "video_url": video_url,
        "results": results,
        "summary": dict(summary)
    }
