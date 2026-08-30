import tensorflow as tf
from pathlib import Path

# Define your image directory and classes
DATA_DIR = Path("data/AIDER_Images")
CLASS_NAMES = ["collapse", "fire", "flood", "normal"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_and_preprocess_image(file_path, label):
    """Loads an image and preprocesses it for ResNet50"""
    # Read image
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    
    # Resize and normalize (ResNet50 expects specific preprocessing)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    
    return img, label

def get_dataset(split="train", validation_split=0.2, seed=42):
    """
    Creates a highly optimized tf.data.Dataset from the folder structure.
    """
    # 1. Get all image paths and labels
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
    
    # 2. Convert to TensorFlow Datasets
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(all_labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))
    
    # 3. Shuffle and split into train/val
    ds = ds.shuffle(buffer_size=len(all_image_paths), seed=seed)
    val_size = int(len(all_image_paths) * validation_split)
    
    val_ds = ds.take(val_size)
    train_ds = ds.skip(val_size)
    
    # 4. Map the preprocessing function and optimize for performance
    train_ds = (train_ds
                .map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(BATCH_SIZE)
                .cache()
                .prefetch(tf.data.AUTOTUNE))
                
    val_ds = (val_ds
              .map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(BATCH_SIZE)
              .prefetch(tf.data.AUTOTUNE))
              
    return train_ds, val_ds

# ==========================================
# TEST IT RIGHT NOW
# ==========================================
if __name__ == "__main__":
    print("🚀 Testing Vision Data Pipeline...")
    train_ds, val_ds = get_dataset()
    
    # Grab one batch to verify shapes
    for images, labels in train_ds.take(1):
        print(f"\n✅ SUCCESS!")
        print(f"Image batch shape: {images.shape}")   # Should be (32, 224, 224, 3)
        print(f"Labels batch shape: {labels.shape}")   # Should be (32,)
        print(f"Classes found: {CLASS_NAMES}")
        break