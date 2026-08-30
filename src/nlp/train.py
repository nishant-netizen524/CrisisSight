import tensorflow as tf
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting Text Encoder Setup & Test (Hugging Face Version)...")

# ==========================================
# 1. LOAD THE TEXT DATA
# ==========================================
csv_path = Path("data/DisasterTweets/train.csv")

if not csv_path.exists():
    raise FileNotFoundError(f"❌ Could not find {csv_path}. Please run the download script first!")

print(f"📂 Loading text data from {csv_path}...")
df = pd.read_csv(csv_path)

# Clean the text (fill NaNs, strip whitespace)
df['text'] = df['text'].fillna("").astype(str).str.strip() #type: ignore

# Take a small batch for testing (first 32 rows)
sample_texts = df['text'].head(32).tolist()

print(f"✅ Loaded {len(sample_texts)} sample texts.")
print(f"Sample text: '{sample_texts[0]}'")

# ==========================================
# 2. BUILD THE TEXT ENCODER (Hugging Face)
# ==========================================
print("\n🧠 Building Text Encoder (Downloading MiniLM from Hugging Face)...")
print("(This is a one-time download of ~80MB)")

# Load a lightweight, fast sentence transformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# ==========================================
# 3. TEST THE ENCODER
# ==========================================
print("\n🧪 Testing encoder on sample texts...")

# Encode the texts (returns numpy array of shape (32, 384))
embeddings_np = model.encode(sample_texts)

# Convert to TensorFlow tensor and project to 256-dim to match Vision Encoder
embeddings_tf = tf.convert_to_tensor(embeddings_np, dtype=tf.float32)

# Create a simple Dense layer to project 384 -> 256
projection_layer = tf.keras.layers.Dense(256, activation='relu', name="text_embedding")
final_embeddings = projection_layer(embeddings_tf)

print(f"✅ SUCCESS!")
print(f"Input: {len(sample_texts)} texts")
print(f"Raw embedding shape: {embeddings_tf.shape}") # (32, 384)
print(f"Final projected shape: {final_embeddings.shape}") # (32, 256)

# ==========================================
# 4. SAVE THE MODEL (FIXED)
# ==========================================
print("\n💾 Saving text encoder...")
Path("models").mkdir(parents=True, exist_ok=True)

# Build a proper Sequential model
projection_model = tf.keras.Sequential()
projection_model.add(tf.keras.layers.Dense(256, activation='relu', input_shape=(384,), name="text_embedding")) #type: ignore

# Transfer the weights from our test layer to the model
projection_model.layers[0].set_weights(projection_layer.get_weights())

# Save it
projection_model.save("models/text_projection_layer.h5")

# Also save the raw embeddings for this batch as a quick test artifact
np.save("models/sample_text_embeddings.npy", embeddings_np)

print("✅ Text projection model saved to models/text_projection_layer.h5")
print("🎉 Person B's Text Pipeline is READY for fusion!")