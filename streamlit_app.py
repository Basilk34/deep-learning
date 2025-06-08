# streamlit_app.py

import streamlit as st
from googleapiclient.discovery import build

# 🔐 إعدادات API
YOUTUBE_API_KEY = "AIzaSyANEG0NbdmV_veIiZHY9cyK-0du_cYmtRk"
REGION_CODE = "SA"  # غيرها حسب الدولة إذا بدك
MAX_RESULTS = 10

# تابع جلب فيديوهات ترند بناءً على موضوع
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

# 🧪 واجهة Streamlit
st.set_page_config(page_title="يوتيوب ترند", layout="centered")
st.title("📺 YouTube Trend Explorer")
st.write("اكتب موضوع وشوف الفيديوهات الترند عليه 👇")

query = st.text_input("🎯 الموضوع (مثلاً: كرة القدم، رمضان، BTS):")

if query:
    with st.spinner("🔍 جاري البحث عن الفيديوهات..."):
        try:
            videos = search_trending_videos(YOUTUBE_API_KEY, query, REGION_CODE, MAX_RESULTS)
            if not videos:
                st.warning("😕 لا يوجد فيديوهات ترند على هذا الموضوع حاليًا.")
            else:
                st.success(f"✅ تم العثور على {len(videos)} فيديو")
                for title, url in videos:
                    st.markdown(f"📌 [{title}]({url})")
        except Exception as e:
            st.error(f"❌ خطأ: {e}")
