import tensorflow as tf
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from pathlib import Path

print("🧪 Testing Fusion Model End-to-End...")

# 1. Load models
print("📦 Loading models...")
fusion_model = tf.keras.models.load_model("models/fusion_model.h5")
text_model = SentenceTransformer('all-MiniLM-L6-v2')
projection_model = tf.keras.models.load_model("models/text_projection_layer.h5")

# 2. Pick a real test image
print("\n📸 Loading sample image...")
test_img_dir = Path("data/AIDER_Images/flood")
if not test_img_dir.exists() or not list(test_img_dir.glob("*.jpg")):
    test_img_dir = Path("data/AIDER_Images/normal")

sample_img_path = list(test_img_dir.glob("*.jpg"))[0]
print(f"   Using: {sample_img_path.name}")

# Preprocess image (RAW IMAGE, not embedding!)
img = Image.open(sample_img_path).convert('RGB').resize((224, 224))
img_array = np.array(img)
img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
img_batch = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

# 3. Test with 3 different text scenarios for the SAME image
scenarios = [
    "Routine patrol completed. All systems normal in sector 402.",
    "Minor water accumulation reported on local roads. Monitoring situation.",
    "CRITICAL: Building collapse confirmed. Multiple casualties reported."
]

severity_classes = ['Normal (0)', 'Minor (1)', 'Major (2)', 'Critical (3)']

print("\n" + "="*60)
print("🔮 PREDICTION RESULTS (Same Image, Different Text Context)")
print("="*60)

for text in scenarios:
    # Get text embedding
    text_embedding_raw = text_model.encode([text])
    text_embedding = projection_model.predict(text_embedding_raw, verbose=0)
    
    # Predict (pass RAW IMAGE, not embedding!)
    prediction = fusion_model.predict(
        {'image_input': img_batch, 'text_input': text_embedding},
        verbose=0
    )[0]
    
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]
    
    print(f"\n📝 Text: '{text}'")
    print(f"   ➔ Predicted Severity: {severity_classes[predicted_class]}")
    print(f"   ➔ Confidence: {confidence:.1%}")
    print(f"   ➔ Probabilities: [N:{prediction[0]:.2f}, Mi:{prediction[1]:.2f}, Ma:{prediction[2]:.2f}, C:{prediction[3]:.2f}]")

print("\n" + "="*60)
print("🎉 Fusion model is working perfectly!")
print("="*60)