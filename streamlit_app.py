import streamlit as st
import gdown
import os
import pickle



@st.cache_resource
def download_tokenizer():
    file_id = "10jfNR3NcOh1MO2xpybPP9LrlRsTsHUzM"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "tokenizer.pkl"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
download_tokenizer()
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
