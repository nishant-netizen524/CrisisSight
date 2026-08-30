import requests
from pathlib import Path

print("🧪 Testing CrisisSight API...")

# API endpoint
API_URL = "http://localhost:8001"  # Change to 8000 if using default port

# Test 1: Health check
print("\n1️⃣ Testing /health endpoint...")
try:
    response = requests.get(f"{API_URL}/health", timeout=5)
    if response.status_code == 200:
        print(f"✅ Health check passed: {response.json()}")
    else:
        print(f"❌ Health check failed: {response.status_code}")
except Exception as e:
    print(f"❌ API not reachable: {e}")
    print("   Make sure the API is running: python api/main.py")
    exit(1)

# Test 2: Prediction endpoint
print("\n2️⃣ Testing /predict endpoint...")

# Find a test image
test_image_path = None
for img_dir in ["data/AIDER_Images/flood", "data/AIDER_Images/fire", "data/AIDER_Images/normal"]:
    img_path = Path(img_dir)
    if img_path.exists():
        images = list(img_path.glob("*.jpg")) + list(img_path.glob("*.png"))
        if images:
            test_image_path = images[0]
            break

if not test_image_path:
    print("❌ No test images found in data/AIDER_Images/")
    exit(1)

print(f"   Using test image: {test_image_path}")

# Prepare the request
with open(test_image_path, 'rb') as f:
    files = {
        'image': (test_image_path.name, f, 'image/jpeg')
    }
    data = {
        'text_report': 'Massive flooding in sector 4, water level rising rapidly, rescue needed'
    }
    
    print("   Sending request to API...")
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            files=files,
            data=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Prediction successful!")
            print(f"   Severity: {result['severity_class']}")
            print(f"   Confidence: {result['confidence']:.1%}")
            print(f"\n📋 Action Plan:")
            print(f"   Action: {result['action_plan']['action']}")
            print(f"   Resources: {', '.join(result['action_plan']['resources'])}")
            print(f"   Reasoning: {result['action_plan']['reasoning']}")
            print(f"   Response Time: {result['action_plan']['estimated_response_time']}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except requests.exceptions.Timeout:
        print(" Request timed out (30s)")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n✅ API test complete!")