import streamlit as st
import numpy as np
import pickle
import gdown
import os

file_id = "10jfNR3NcOh1MO2xpybPP9LrlRsTsHUzM"
url = f"https://drive.google.com/uc?id={file_id}"
output = "tokenizerrr.pkl"

try:
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    with open(output, "rb") as f:
        tokenizer = pickle.load(f)
        st.success("✅ تم تحميل التوكينايزر")
except Exception as e:
    st.error(f"❌ حدث خطأ أثناء تحميل التوكينايزر: {e}")

