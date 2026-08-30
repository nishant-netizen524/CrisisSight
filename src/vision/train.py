import tensorflow as tf
from tensorflow.keras import layers
from pathlib import Path

print("🚀 Starting Vision Encoder Training (Self-Contained Version)...")

# ==========================================
# 1. DEFINE THE MODEL DIRECTLY HERE
# ==========================================
def build_vision_encoder(input_shape=(224, 224, 3)):
    """ResNet50 feature extractor → 256-dim embedding"""
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
        pooling='avg'
    )
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = base_model(x, training=False)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(256, activation='relu', name="vision_embedding")(x)
    
    return tf.keras.Model(inputs, outputs, name="VisionEncoder")

# ==========================================
# 2. DEFINE DATA LOADER DIRECTLY HERE
# ==========================================
DATA_DIR = Path("data/AIDER_Images")  # Adjusted path
CLASS_NAMES = ["collapse", "fire", "flood", "normal"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_and_preprocess_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, label

def get_dataset(validation_split=0.2, seed=42):
    all_image_paths = []
    all_labels = []
    
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = DATA_DIR / class_name
        if not class_dir.exists():
            print(f"⚠️ Warning: Folder {class_dir} not found. Skipping.")
            continue
        paths = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        all_image_paths.extend([str(p) for p in paths])
        all_labels.extend([label] * len(paths))
    
    print(f"✅ Found {len(all_image_paths)} total images across {len(CLASS_NAMES)} classes.")
    
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(all_labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))
    ds = ds.shuffle(buffer_size=len(all_image_paths), seed=seed)
    
    val_size = int(len(all_image_paths) * validation_split)
    val_ds = ds.take(val_size)
    train_ds = ds.skip(val_size)
    
    train_ds = (train_ds
                .map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))
    
    val_ds = (val_ds
              .map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(BATCH_SIZE)
              .prefetch(tf.data.AUTOTUNE))
    
    return train_ds, val_ds

# ==========================================
# 3. MAIN TRAINING CODE
# ==========================================
if __name__ == "__main__":
    # Load data
    train_ds, val_ds = get_dataset()

    
    
    # Build vision encoder
    vision_encoder = build_vision_encoder()
    
    # Add classification head
    inputs = vision_encoder.input
    x = vision_encoder.output
    outputs = layers.Dense(len(CLASS_NAMES), activation='softmax', name="classifier")(x)
    full_model = tf.keras.Model(inputs, outputs, name="VisionClassifier")
    
    full_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n📊 Model Summary:")
    full_model.summary()
    
    # Train
    print("\n🏋️ Starting training...")
    history = full_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        verbose=1
    )
    
    # Save models
    print("\n💾 Saving models...")
    Path("models").mkdir(parents=True, exist_ok=True)
    
    full_model.save("models/vision_classifier.h5")
    vision_encoder.save("models/vision_encoder.h5")
    
    print("✅ Models saved!")
    final_acc = history.history['val_accuracy'][-1]
    print(f"\n🎯 Final Validation Accuracy: {final_acc:.2%}")
    print("✅ Vision Encoder Training Complete!")