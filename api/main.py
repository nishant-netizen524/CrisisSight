import sys
from pathlib import Path

# CRITICAL: Add project root to path BEFORE importing agent
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
import io

from agent.crisis_agent import CrisisAgent

print("🚀 Starting CrisisSight API...")

# ==========================================
# 1. INITIALIZE FASTAPI APP
# ==========================================
app = FastAPI(
    title="CrisisSight API",
    description="Multimodal AI for Disaster Damage Assessment",
    version="1.0.0"
)

# Enable CORS so Streamlit frontend can talk to the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. LOAD MODELS ONCE AT STARTUP
# ==========================================
print("📦 Loading AI models (this takes ~30 seconds on first run)...")

# Import SentenceTransformer here to avoid loading it twice
from sentence_transformers import SentenceTransformer

# Load all models into global variables
fusion_model = None
text_model = None
projection_model = None
crisis_agent = None

@app.on_event("startup")
async def load_models():
    global fusion_model, text_model, projection_model, crisis_agent
    
    try:
        print("  → Loading fusion model...")
        fusion_model = tf.keras.models.load_model("models/fusion_model.h5")
        
        print("  → Loading text encoder (MiniLM)...")
        text_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("  → Loading text projection layer...")
        projection_model = tf.keras.models.load_model("models/text_projection_layer.h5")
        
        print("  → Initializing LLM agent (Groq)...")
        crisis_agent = CrisisAgent()
        
        print("✅ All models loaded successfully!")
    except Exception as e:
        print(f"❌ ERROR loading models: {e}")
        raise

# ==========================================
# 3. RESPONSE SCHEMAS
# ==========================================
class ActionPlan(BaseModel):
    severity: str
    action: str
    resources: list
    reasoning: str
    estimated_response_time: str

class PredictionResponse(BaseModel):
    severity_class: str
    severity_level: int
    confidence: float
    probabilities: dict
    action_plan: dict
    message: str

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================
SEVERITY_CLASSES = ['Normal', 'Minor', 'Major', 'Critical']

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert uploaded image bytes to model-ready format"""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

def encode_text(text: str) -> np.ndarray:
    """Convert text to 256-dim embedding"""
    raw_embedding = text_model.encode([text])   #type: ignore
    return projection_model.predict(raw_embedding, verbose=0)  #type: ignore

# ==========================================
# 5. MAIN ENDPOINTS
# ==========================================
@app.get("/")
async def root():
    return {
        "service": "CrisisSight API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST) - Analyze image + text",
            "health": "/health (GET) - Check service status"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": fusion_model is not None,
        "agent_ready": crisis_agent is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(..., description="Disaster image (JPG/PNG)"),
    text_report: str = Form(..., description="Ground report or description")
):
    """
    Analyze a disaster image with accompanying text report.
    Returns severity prediction + AI-generated action plan.
    """
    # 1. Validate inputs
    if not text_report or len(text_report.strip()) < 3:
        raise HTTPException(status_code=400, detail="Text report is too short.")
    
    # 2. Read and preprocess image
    image_bytes = await image.read()
    img_batch = preprocess_image(image_bytes)
    
    # 3. Encode text
    text_embedding = encode_text(text_report)
    
    # 4. Run fusion model
    prediction = fusion_model.predict(    #type: ignore
        {'image_input': img_batch, 'text_input': text_embedding},
        verbose=0
    )[0]  
    
    predicted_idx = int(np.argmax(prediction))
    confidence = float(prediction[predicted_idx])
    severity_class = SEVERITY_CLASSES[predicted_idx]
    
    probabilities = {
        SEVERITY_CLASSES[i]: round(float(prediction[i]), 4)
        for i in range(len(SEVERITY_CLASSES))
    }
    
    # 5. Generate action plan using LLM agent
    try:
        action_plan = crisis_agent.generate_report(  #type: ignore
            severity=severity_class,
            image_context=f"Image analyzed. Model predicts {severity_class} severity with {confidence:.1%} confidence.",
            text_report=text_report
        )
    except Exception as e:
        print(f"⚠️ LLM agent error: {e}")
        action_plan = {
            "severity": severity_class,
            "action": "Manual assessment required",
            "resources": ["Emergency response team"],
            "reasoning": f"AI agent unavailable: {str(e)}",
            "estimated_response_time": "ASAP"
        }
    
    # 6. Build response
    return PredictionResponse(
        severity_class=severity_class,
        severity_level=predicted_idx,
        confidence=round(confidence, 4),
        probabilities=probabilities,
        action_plan=action_plan,
        message=f"Assessment complete. Severity: {severity_class} ({confidence:.1%} confidence)."
    )

# ==========================================
# 6. RUN SERVER
# ==========================================
if __name__ == "__main__":
    import uvicorn
    print("\n🌐 Starting server on http://localhost:8001")
    print("📚 API docs available at http://localhost:8001/docs")
    uvicorn.run(app, host="0.0.0.0", port=8001)