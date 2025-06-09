import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from utils.face_detect import detect_face

general_model = load_model("models/general_model.h5")
face_model = load_model("models/face_model.h5")
labels = ['سلبي', 'محايد', 'إيجابي']

def classify_image(image_path):
    image = cv2.imread(image_path)
    face = detect_face(image)
    if face is not None:
        resized = cv2.resize(face, (224, 224))
        model = face_model
    else:
        resized = cv2.resize(image, (224, 224))
        model = general_model
    arr = img_to_array(resized) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr)[0]
    return labels[np.argmax(pred)], pred
