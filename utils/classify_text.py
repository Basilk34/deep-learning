import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

labels = ['neutral', 'negative', 'positive']

def classify_text(text, max_len=100):
    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    model = load_model("models/text_model.h5")
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded)[0]
    return labels[np.argmax(pred)], pred
