
import gdown
import os

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Google Drive file links and output names
files = {
    "https://drive.google.com/uc?id=18e9JcIpWWkRke1Rh2fEu_B1u6pbpzBh0": "models/general_model.h5",
    "https://drive.google.com/uc?id=1QiS1oEYxnIbj3ykZ-OmfqmUj7u0yHYN3": "models/face_model.h5",
    "https://drive.google.com/uc?id=1EnKgCo20_-lMkhsJPsdvPiGe6jK7M3EJ": "models/text_model.h5",
    "https://drive.google.com/uc?id=1vsrmpQ1XrOiboH8ZrlTraYp3uuLfiPET": "models/tokenizer.pkl"
}

# Download each file
for url, output in files.items():
    print(f"📥 Downloading {output} ...")
    gdown.download(url, output, quiet=False)
