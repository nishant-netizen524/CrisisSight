import pandas as pd
import os

print("🔧 Creating sample disaster tweets dataset...")

os.makedirs("data/DisasterTweets", exist_ok=True)

# Create 100 realistic sample tweets
data = []

# Real disaster tweets (target = 1)
disaster_tweets = [
    "Massive flooding in Sector 4, water level rising rapidly",
    "Building collapsed, people trapped inside, need rescue immediately",
    "Fire spreading to adjacent structures, evacuate now",
    "Roads completely submerged, rescue boats needed",
    "Buildings flooded up to 2nd floor, urgent help required",
    "Smoke visible from 5km away, building on fire",
    "Structural damage everywhere, immediate rescue needed",
    "Rubble blocking main road, ambulances can't pass",
    "Evacuation in progress, stay away from the area",
    "Water level rising fast, people on rooftops waiting for rescue"
]

# Non-disaster tweets (target = 0)
normal_tweets = [
    "Beautiful sunny day at the beach today",
    "Just finished my morning workout, feeling great",
    "New restaurant opened downtown, amazing food",
    "Traffic is terrible this morning, late for work",
    "Watching the game tonight, go team!",
    "Just adopted a puppy, so cute",
    "Perfect weather for a hike this weekend",
    "New phone update is amazing",
    "Coffee shop has the best latte in town",
    "Movie night with friends, so much fun"
]

# Generate 100 samples (50 disaster, 50 normal)
for i in range(50):
    data.append({
        'id': i,
        'keyword': 'disaster' if i % 2 == 0 else '',
        'location': '',
        'text': disaster_tweets[i % len(disaster_tweets)] + f" #{i}",
        'target': 1
    })
    data.append({
        'id': i + 50,
        'keyword': '',
        'location': '',
        'text': normal_tweets[i % len(normal_tweets)] + f" #{i+50}",
        'target': 0
    })

df = pd.DataFrame(data)
df.to_csv("data/DisasterTweets/train.csv", index=False)

print(f"✅ Created sample dataset with {len(df)} tweets")
print(f"Saved to: data/DisasterTweets/train.csv")
print("\n📊 Preview:")
print(df.head())