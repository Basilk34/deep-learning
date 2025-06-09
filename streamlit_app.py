import pickle
import streamlit as st
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("arabic_sentiment_model_clean.h5", compile=False)

# Load the tokenizer
with open("tok.pkl", "rb") as f:
    tokenizer = pickle.load(f)

st.success("✅ تم تحميل الموديل والتوكينايزر بنجاح.")

