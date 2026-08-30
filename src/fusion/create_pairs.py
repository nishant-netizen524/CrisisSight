import pandas as pd
import numpy as np
from pathlib import Path
import random

print("🔗 Creating CLEAN, diverse image-text pairs for fusion training...")

DATA_DIR = Path("data/AIDER_Images")
CLASS_NAMES = ["collapse", "fire", "flood", "normal"]

# 1. Define diverse templates for each severity level
# This prevents exact string matching leakage while keeping the signal clear
TEMPLATES = {
    0: [  # Normal
        "Routine patrol completed. All systems normal in sector {id}.",
        "No anomalies detected. Area is clear and safe.",
        "Standard daily report: no incidents or hazards observed.",
        "Traffic flowing normally, no emergency services required."
    ],
    1: [  # Minor (e.g., minor flood or small issue)
        "Minor water accumulation reported on local roads. Monitoring situation.",
        "Small debris on roadway. Caution advised for local traffic.",
        "Minor structural wear observed. No immediate threat to public.",
        "Localized minor flooding, no evacuations necessary at this time."
],
    2: [  # Major (e.g., significant flood or damage)
        "Significant structural damage reported. Road is completely blocked.",
        "Major flooding in the area. Evacuation of nearby residents advised.",
        "Severe weather impact: multiple buildings sustained heavy damage.",
        "Emergency crews dispatched to handle major infrastructure failure."
    ],
    3: [  # Critical (e.g., fire, collapse)
        "CRITICAL: Building collapse confirmed. Multiple casualties reported.",
        "URGENT: Massive fire spreading rapidly. Immediate evacuation required.",
        "Life-threatening situation: trapped civilians, need immediate rescue.",
        "CATASTROPHIC failure: entire sector compromised, all units respond."
    ]
}

# 2. Gather image data
image_data = []
for label, class_name in enumerate(CLASS_NAMES):
    class_dir = DATA_DIR / class_name
    if class_dir.exists():
        paths = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        for p in paths:
            image_data.append({'image_path': str(p), 'class_name': class_name})

random.seed(42)
random.shuffle(image_data)

# 3. Map image class to a base severity, then generate diverse text
def get_severity_and_text(class_name, template_id):
    if class_name == 'normal':
        severity = 0
    elif class_name == 'flood':
        severity = 2  # Major
    elif class_name in ['fire', 'collapse']:
        severity = 3  # Critical
    else:
        severity = 1  # Minor fallback
        
    # Pick a random template for this severity and fill in a random ID
    template = random.choice(TEMPLATES[severity])
    text = template.format(id=random.randint(100, 999))
    return severity, text

# 4. Create Train and Val sets (80/20 split of IMAGES, ensuring diverse text)
split_idx = int(len(image_data) * 0.8)
train_images = image_data[:split_idx]
val_images = image_data[split_idx:]

train_pairs = []
for i, img in enumerate(train_images):
    severity, text = get_severity_and_text(img['class_name'], i)
    train_pairs.append({'image_path': img['image_path'], 'text': text, 'severity': severity})

val_pairs = []
for i, img in enumerate(val_images):
    severity, text = get_severity_and_text(img['class_name'], i + 1000) # Offset ID to ensure different text
    val_pairs.append({'image_path': img['image_path'], 'text': text, 'severity': severity})

# 5. Save
train_df = pd.DataFrame(train_pairs)
val_df = pd.DataFrame(val_pairs)

train_df.to_csv("data/fusion_train_pairs.csv", index=False)
val_df.to_csv("data/fusion_val_pairs.csv", index=False)

print(f"✅ Created {len(train_df)} train pairs and {len(val_df)} val pairs")
print("\n📊 Train severity distribution:")
print(train_df['severity'].value_counts().sort_index())
print("\n📊 Val severity distribution:")
print(val_df['severity'].value_counts().sort_index())
print("\n💾 Saved to data/fusion_train_pairs.csv and data/fusion_val_pairs.csv")