import streamlit as st

st.title('Basil kanaan')

st.write('deep learning ')
import streamlit as st

# مجموعة منشورات (قاعدة بيانات مصغرة)
posts = [
    "I love this product!",
    "Worst experience ever.",
    "Totally fine and average.",
    "Amazing work from the team.",
    "Horrible service.",
    "Not good, not bad."
]

st.title("🔎 بحث وتحليل التوجه")

# خانة البحث
query = st.text_input("🔍 ابحث عن منشور يحتوي كلمات معينة")

# عرض النتائج المطابقة
if query:
    results = [p for p in posts if query.lower() in p.lower()]
    
    if results:
        st.write("### النتائج:")
        for res in results:
            st.write(f"- {res}")
            # يمكن هنا تحليل التوجه للنص مباشرة
    else:
        st.warning("لا توجد نتائج مطابقة.")
