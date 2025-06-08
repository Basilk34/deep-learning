# streamlit_app.py

import streamlit as st

st.set_page_config(page_title="مشروعي", layout="centered")

st.title("🌟 أهلاً وسهلاً بك يا تاج راسي هنودي 👑")
st.subheader("🚀 هذا مشروع Streamlit سريع لتجربة التشغيل")
st.write("✅ التطبيق يعمل بنجاح! يمكنك الآن البدء بإضافة الموديل، تحليل التعليقات، أو أي مكونات أخرى.")

st.markdown("---")
st.info("💡 إذا وصلتك هذه الرسالة، فهذا يعني أن Streamlit Cloud يعمل 100%.")

st.markdown("### 🧪 اختبر إدخال بسيط:")
user_input = st.text_input("✍️ اكتب أي شيء هنا:")
if user_input:
    st.success(f"✅ تم استلام: {user_input}")
