"""Download datasets used by CrisisSight.

- Disaster Tweets (Kaggle) -> data/DisasterTweets/  (via kagglehub, already in requirements)
- AIDER images: manual download from Zenodo (license requires acceptance);
  this script prints instructions and checks the target folder.
"""
from pathlib import Path

DATA = Path("data")


def download_tweets():
    try:
        try:
            import kagglehub
        except ImportError as e:
            raise RuntimeError(
                "kagglehub is not installed. Install it with: pip install kagglehub"
            ) from e
        path = kagglehub.dataset_download("philculliton/nlp-getting-started-tutorial")
        target = DATA / "DisasterTweets"
        target.mkdir(parents=True, exist_ok=True)
        import shutil
        for f in Path(path).glob("*"):
            shutil.copy(f, target / f.name)
        print(f"✅ Disaster Tweets -> {target}")
    except Exception as e:
        print(f"❌ Kaggle download failed: {e}")
        print("   Fallback: download manually from kaggle.com and put train.csv in data/DisasterTweets/")


def check_aider():
    target = DATA / "AIDER_Images"
    if target.exists() and any(target.iterdir()):
        print(f"✅ AIDER images found at {target}")
    else:
         
        print("⚠️  AIDER dataset not found.")
        print("   1. Download from Zenodo: https://zenodo.org/records/7588286 (AIDER)")
        print("   2. Place images in data/AIDER_Images/{collapse,fire,flood,normal}/")


if __name__ == "__main__":
    download_tweets()
    check_aider()