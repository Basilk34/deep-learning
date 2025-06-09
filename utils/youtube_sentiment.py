import re
import numpy as np
import random
from collections import Counter
from googleapiclient.discovery import build
from youtube_comment_downloader import YoutubeCommentDownloader
from utils.classify_text import classify_text

YOUTUBE_API_KEY = "AIzaSyANEG0NbdmV_veIiZHY9cyK-0du_cYmtRk"

def is_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

def get_trending_videos(api_key, query=None, region='SA', max_results=5):
    youtube = build('youtube', 'v3', developerKey=api_key)
    if query:
        request = youtube.search().list(q=query, part='snippet', type='video', maxResults=max_results)
    else:
        request = youtube.videos().list(part='snippet', chart='mostPopular', regionCode=region, maxResults=max_results)
    response = request.execute()
    return [(item['id']['videoId'] if 'videoId' in item['id'] else item['id'], item['snippet']['title']) for item in response.get('items', [])]

def fetch_arabic_comments(video_id, max_comments=50):
    downloader = YoutubeCommentDownloader()
    raw_comments = []
    for comment in downloader.get_comments_from_url(f"https://www.youtube.com/watch?v={video_id}", sort_by=1):
        text = comment['text']
        if is_arabic(text):
            votes = int(comment.get("votes", "0").replace("K", "000").replace(".", ""))
            raw_comments.append({"text": text, "votes": votes})
        if len(raw_comments) >= max_comments:
            break
    return [c["text"] for c in sorted(raw_comments, key=lambda x: x["votes"], reverse=True)]

def analyze_comments(comments):
    results = [(*[comment], *classify_text(comment)) for comment in comments]
    summary = Counter([r[1] for r in results])
    return results, summary
