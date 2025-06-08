
import gdown
import os

# Google Drive رابط التوكينايزر
file_id = "10jfNR3NcOh1MO2xpybPP9LrlRsTsHUzM"
url = f"https://drive.google.com/uc?id={file_id}"
output = "tokenizer.pkl"

# تحميل الملف فقط إذا لم يكن موجود
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)
