import streamlit as st
import snscrape.modules.twitter as sntwitter
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
# ------------------------
# تحميل النموذج والتوكنيزر
# ------------------------
model = tf.keras.models.load_model("text_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

label_map = ["سلبي", "محايد", "إيجابي"]

# ------------------------
# إعداد Streamlit
# ------------------------
st.set_page_config(page_title="تحليل التوجه", layout="wide")
st.title("📊 تطبيق تحليل التوجه - Basil Kanaan")

# ------------------------
# تبويبات Streamlit
# ------------------------
tab1, tab2, tab3 = st.tabs([
    "🔍 تحليل نص بحثي",
    "🐦 تحليل تغريدات Twitter",
    "📂 بحث في منشورات وهمية"
])

# ------------------------
# التبويب 1: تحليل نص على شكل بحث
# ------------------------
with tab1:
    st.subheader("🔍 أدخل نصًا لتحليله مباشرة:")
    search_text = st.text_input("✍️ اكتب جملة أو تعليق:")
    if search_text:
        seq = tokenizer.texts_to_sequences([search_text])
        padded = pad_sequences(seq, maxlen=10)
        pred = model.predict(padded, verbose=0)
        label = label_map[np.argmax(pred)]

        st.markdown(f"""
        ---
        📝 **النص:** {search_text}  
        🧠 **التوجه:** `{label}`
        """)

# ------------------------
# التبويب 2: تحليل تغريدات Twitter
# ------------------------
with tab2:
    st.subheader("🐦 أدخل كلمة مفتاحية:")
    keyword = st.text_input("🔑 مثال: الذكاء الاصطناعي")
    tweet_count = st.slider("🔢 عدد التغريدات", 5, 50, 10)

    if st.button("تحليل التغريدات"):
        if not keyword:
            st.warning("يرجى إدخال كلمة.")
        else:
            st.info("📡 جارٍ جلب وتحليل التغريدات...")
            tweets = []
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(keyword).get_items()):
                if i >= tweet_count:
                    break
                tweets.append(tweet.content)

            for tweet in tweets:
                seq = tokenizer.texts_to_sequences([tweet])
                padded = pad_sequences(seq, maxlen=10)
                pred = model.predict(padded, verbose=0)
                label = label_map[np.argmax(pred)]
                st.markdown(f"""
                ---
                📝 **النص:** {tweet}  
                🧠 **التوجه:** `{label}`
                """)

# ------------------------
# التبويب 3: بحث وتحليل منشورات وهمية
# ------------------------
with tab3:
    st.subheader("📂 منشورات تجريبية")

    posts = [
        "I love this product!",
        "Worst experience ever.",
        "Totally fine and average.",
        "Amazing work from the team.",
        "Horrible service.",
        "Not good, not bad."
    ]

    query = st.text_input("🔍 ابحث داخل المنشورات:")
    if query:
        results = [p for p in posts if query.lower() in p.lower()]
        if results:
            st.write("### ✅ النتائج:")
            for res in results:
                st.write(f"- {res}")
                seq = tokenizer.texts_to_sequences([res])
                padded = pad_sequences(seq, maxlen=10)
                pred = model.predict(padded, verbose=0)
                label = label_map[np.argmax(pred)]
                st.write(f"🧠 التوجه: `{label}`")
        else:
            st.warning("🚫 لا توجد نتائج مطابقة.")
