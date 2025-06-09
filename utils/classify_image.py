import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# تحميل الموديل العام فقط
model = load_model("models/general_model.h5")

# ترتيب التصنيفات حسب ما قلت:
labels = ['neutral', 'negative', 'positive']

def classify_image(image_path):
    image = cv2.imread(image_path)
    resized = cv2.resize(image, (224, 224))

    arr = img_to_array(resized) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)[0]
    return labels[np.argmax(pred)], pred
