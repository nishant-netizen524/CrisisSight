import tensorflow as tf
from pathlib import Path

DATA_DIR = Path("data/AIDER_Images")
CLASS_NAMES = ["collapse", "fire", "flood", "normal"]

print("🔍 Scanning for TensorFlow-incompatible images...")

bad_files = []

for class_name in CLASS_NAMES:
    class_dir = DATA_DIR / class_name
    if not class_dir.exists():
        continue
    
    print(f"Checking {class_name}...")
    # Check jpg, jpeg, png (case-insensitive)
    paths = (list(class_dir.glob("*.jpg")) + 
             list(class_dir.glob("*.jpeg")) + 
             list(class_dir.glob("*.png")) +
             list(class_dir.glob("*.JPG")) +
             list(class_dir.glob("*.PNG")))
    
    for p in paths:
        try:
            # This is the EXACT operation that crashes during training
            img_bytes = tf.io.read_file(str(p))
            img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        except Exception as e:
            print(f"  ❌ TF CANNOT READ: {p.name}")
            bad_files.append(p)

print(f"\n{'='*50}")
if bad_files:
    print(f"🚨 Found {len(bad_files)} files that crash TensorFlow!")
    print("Deleting them now...")
    for p in bad_files:
        p.unlink()
        print(f"  🗑️ Deleted: {p.name}")
    print("✅ Cleanup complete! You can now run train.py safely.")
else:
    print("✅ All images are perfectly compatible with TensorFlow!")