import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sentence_transformers import SentenceTransformer

print("🚀 Starting Multimodal Fusion Training...")

# ==========================================
# 1. LOAD THE PAIRED DATA
# ==========================================
# NEW CODE (separate train/val files)
train_df = pd.read_csv("data/fusion_train_pairs.csv")
val_df = pd.read_csv("data/fusion_val_pairs.csv")

print(f"✅ Train: {len(train_df)} | Val: {len(val_df)}")

# ==========================================
# 2. LOAD PRE-TRAINED ENCODERS
# ==========================================
print("\n🧠 Loading pre-trained encoders...")

# Load Vision Encoder
vision_encoder = tf.keras.models.load_model("models/vision_encoder.h5")
vision_encoder.trainable = False  # Freeze it
print("✅ Vision encoder loaded (frozen)")

# Load Text Encoder (MiniLM)
text_model = SentenceTransformer('all-MiniLM-L6-v2')
projection_model = tf.keras.models.load_model("models/text_projection_layer.h5")
projection_model.trainable = False  # Freeze it
print("✅ Text encoder loaded (frozen)")

# ==========================================
# 3. CREATE DATA GENERATOR
# ==========================================
def create_dataset(df, batch_size=16):
    """Creates a tf.data.Dataset from the pairs dataframe"""
    
    def load_and_preprocess_image(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (224, 224))
        img = tf.keras.applications.resnet50.preprocess_input(img)
        return img
    
    # Process images
    image_paths = df['image_path'].tolist()
    images = []
    for path in image_paths:
        try:
            img = load_and_preprocess_image(path)
            images.append(img.numpy())
        except:
            # If image fails to load, use a black image
            images.append(np.zeros((224, 224, 3)))
    
    images = np.array(images)
    
    # Process texts with MiniLM
    texts = df['text'].tolist()
    text_embeddings_raw = text_model.encode(texts, show_progress_bar=False)
    
    # Project to 256-dim
    text_embeddings = projection_model.predict(text_embeddings_raw, verbose=0)
    
    # Get labels
    labels = df['severity'].values
    
    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        ({'image_input': images, 'text_input': text_embeddings}, labels) #type: ignore
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

train_ds = create_dataset(train_df, batch_size=16)
val_ds = create_dataset(val_df, batch_size=16)

print(f"✅ Datasets created")

# ==========================================
# 4. BUILD THE FUSION MODEL
# ==========================================
print("\n🏗️ Building fusion model...")

# Image input branch
image_input = tf.keras.Input(shape=(224, 224, 3), name='image_input')
img_embedding = vision_encoder(image_input, training=False)  # 256-dim

# Text input branch (already 256-dim from projection)
text_input = tf.keras.Input(shape=(256,), name='text_input')

# Fuse them
merged = tf.keras.layers.Concatenate()([img_embedding, text_input])  # 512-dim
x = tf.keras.layers.Dense(256, activation='relu')(merged)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
output = tf.keras.layers.Dense(4, activation='softmax', name='severity_output')(x)

fusion_model = tf.keras.Model(
    inputs=[image_input, text_input],
    outputs=output,
    name='CrisisSight_Fusion'
)

fusion_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📊 Fusion Model Summary:")
fusion_model.summary()

# ==========================================
# 5. TRAIN THE MODEL
# ==========================================
print("\n🏋️ Starting fusion training...")

history = fusion_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    verbose=1
)

# ==========================================
# 6. SAVE THE FUSION MODEL
# ==========================================
print("\n💾 Saving fusion model...")
Path("models").mkdir(parents=True, exist_ok=True)
fusion_model.save("models/fusion_model.h5")

print("✅ Fusion model saved to models/fusion_model.h5")

# Print final metrics
final_acc = history.history['val_accuracy'][-1]
print(f"\n🎯 Final Validation Accuracy: {final_acc:.2%}")
print("🎉 MULTIMODAL FUSION COMPLETE!")