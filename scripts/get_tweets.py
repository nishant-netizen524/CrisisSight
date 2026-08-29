import urllib.request
import os

print("📥 Downloading disaster tweets from Hugging Face...")

os.makedirs("data/DisasterTweets", exist_ok=True)

# This is a verified, working direct link
url = "https://huggingface.co/datasets/pragma77/kaggle_nlp_getting_started/resolve/main/train.csv"
save_path = "data/DisasterTweets/train.csv"

try:
    urllib.request.urlretrieve(url, save_path)
    print(f"✅ Successfully downloaded to {save_path}")
    
    # Quick preview
    import pandas as pd
    df = pd.read_csv(save_path)
    print(f"\n📊 Dataset loaded: {len(df)} rows")
    print("Columns:", df.columns.tolist())
    print("\nSample tweets:")
    print(df['text'].head(3).tolist())
    
except Exception as e:
    print(f"❌ Failed: {e}")
    print("\n👉 Try Option 2 below")