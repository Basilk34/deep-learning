# streamlit_app.py

import streamlit as st
from googleapiclient.discovery import build
import random

# إعدادات الـ API
YOUTUBE_API_KEY = "YOUR_YOUTUBE_API_KEY"
MAX_RESULTS = 10
REGION_CODE = "SA"  # يمكنك تغييره حسب البلد (مثلاً "EG" لمصر أو "JO" للأردن)

# تابع للبحث في ترندات يوتيوب حسب موضوع
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
        results.append((title, url))
    return results

# ========================
# 🌐 واجهة Streamlit
# ========================

st.title("🔍 YouTube Trend Explorer (بالعربي)")
st.write("اكتب موضوع معين وشوف شو الفيديوهات الترند عليه من YouTube")

query = st.text_input("🎯 اكتب الموضوع (مثلاً: كرة القدم، رمضان، BTS، الخ...):")

if query:
    with st.spinner("جاري جلب الفيديوهات..."):
        try:
            videos = search_trending_videos(YOUTUBE_API_KEY, query, REGION_CODE, MAX_RESULTS)
            if not videos:
                st.warning("😕 لا يوجد فيديوهات ترند على هذا الموضوع حاليًا.")
            else:
                st.success(f"📈 عدد الفيديوهات: {len(videos)}")
                for title, url in videos:
                    st.markdown(f"- [{title}]({url})")
        except Exception as e:
            st.error(f"❌ حصل خطأ: {e}")
