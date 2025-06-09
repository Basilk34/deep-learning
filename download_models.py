import gdown
import os

model_links = {
    "models/general_model.h5": "https://drive.google.com/uc?id=18e9JcIpWWkRke1Rh2fEu_B1u6pbpzBh0",
    "models/text_model.h5": "https://drive.google.com/uc?id=1EnKgCo20_-lMkhsJPsdvPiGe6jK7M3EJ",
    "models/tokenizer.pkl": "https://drive.google.com/uc?id=1vsrmpQ1XrOiboH8ZrlTraYp3uuLfiPET"
}

os.makedirs("models", exist_ok=True)

for path, url in model_links.items():
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        gdown.download(url, path, quiet=False)
