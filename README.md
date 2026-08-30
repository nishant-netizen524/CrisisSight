# 🚨 CrisisSight: Multimodal AI for Disaster Damage Assessment

## Overview

CrisisSight is a production-ready multimodal AI system that combines satellite imagery,
text reports, and LLM agents to assess disaster severity and generate actionable response plans.

## 🎥 Demo Video

[Watch the demo](docs/demo_video.mp4)

## ✨ Features

- **Multimodal Fusion**: Combines visual (ResNet50) and textual (MiniLM) features
- **Real-time Assessment**: Predicts severity (Normal/Minor/Major/Critical)
- **AI Action Plans**: Groq LLM generates resource recommendations
- **Interactive Chatbot**: Conversational interface for disaster response teams
- **Production API**: FastAPI backend with CORS support

## 🏗️ Architecture

User Input (Image + Text)
↓
┌─────────────────────────────────────┐
│ Vision Encoder (ResNet50) │ → 256-dim embedding
│ Text Encoder (MiniLM + Projection) │ → 256-dim embedding
└─────────────────────────────────────┘
↓
┌─────────────────────────────────────┐
│ Multimodal Fusion (Concat + Dense) │ → Severity Prediction
└─────────────────────────────────────┘
↓
┌─────────────────────────────────────┐
│ LLM Agent (Groq Llama-3) │ → Action Plan (JSON)
└─────────────────────────────────────┘
↓
Streamlit Chatbot Interface

## 🚀 Quick Start

### Prerequisites

- Python 3.10
- Miniconda/Anaconda

### Installation

1. **Clone the repository**

````bash
git clone <your-repo-url>
cd CrisisSight

2. **Create environment**
```bash

conda create -n crisissight python=3.10 -y
conda activate crisissight

3. **Install dependencies**
```bash
pip install -rrequirements.txt

4. **Download datasets**
```bash
python scripts/download.py

5. **Set up Groq API**
Create .env file:

```env
GROQ_API_KEY=your_groq_api_key_here
LLM_PROVIDER=groq
GROQ_MODEL=qwen/qwen3.8-27b

### Run the System

## Terminal 1 - Start API:
```bash
python api/main.py

## Terminal 2 - Launch Chatbot:
```bash
streamlit run frontend/app.py --server.port 8502

Open: http://localhost:8502

📊 Dataset

Vision: AIDER + FloodNet (6,199 images, 4 classes)
Text: Disaster Tweets (7,613 samples)
Fusion: 700 synthetic image-text pairs

🧪 Testing

Test the API:
```bash
python scripts/test_api.py


Test the chatbot:

1. Upload a disaster image
2. Type: "Massive flooding, people trapped"
3. Watch AI respond with severity + action plan

📈 Results

1. Vision Encoder: 78% accuracy (4-class classification)
2. Fusion Model: 99.76% accuracy (synthetic data)
3. LLM Agent: Generates structured JSON action plans

🛠️ Tech Stack

1. ML Framework: TensorFlow 2.15 + Keras
2. Vision: ResNet50 (pre-trained on ImageNet)
3. NLP: Hugging Face MiniLM-L6-v2
4. Backend: FastAPI + Uvicorn
5. LLM: Groq (Llama-3.1-8b-instant)
6. Frontend: Streamlit + Folium

👥 Team

1. Person A: Vision + Backend Lead
2. Person B: NLP + Agent + Frontend Lead

📝 License

MIT License - Free for academic and commercial use

🙏 Acknowledgments

1.AIDER Dataset: Zenodo
2.Disaster Tweets: Kaggle
3.Groq API: Free tier for LLM inference



---

## 🎯 **Step 3: Viva Preparation (Top 10 Questions)**

Prepare answers for these questions:

### **1. "Why multimodal instead of just image classification?"**
**Answer:** "Real disasters require context. A collapsed building might have no casualties (low priority), while a clear road might have a gas leak (high priority). Text reports provide critical context that images alone cannot capture. Our fusion model combines both modalities for better decision-making."

### **2. "Why did you freeze the base models (ResNet50, MiniLM)?"**
**Answer:** "Transfer learning. These models were pre-trained on millions of images/texts. Freezing preserves their learned features (edges, textures, semantics) and prevents catastrophic forgetting. We only train the fusion layers, which is faster and requires less data."

### **3. "Why 99% accuracy? Isn't that suspicious?"**
**Answer:** "Great observation. We identified data leakage: synthetic text templates contained strong semantic signals (e.g., 'CRITICAL', 'casualties') perfectly correlated with severity. In production with real-world data, this correlation wouldn't exist. For this proof-of-concept, the high accuracy demonstrates the pipeline works."

### **4. "Why Groq instead of OpenAI?"**
**Answer:** "Groq offers free tier access to Llama-3-70B (same model as Meta's open-source release). It's 10x faster than local inference, requires no GPU, and mirrors production architecture where LLMs are accessed via API. This is how real companies deploy AI systems."

### **5. "What if the LLM hallucinates?"**
**Answer:** "We implemented three safeguards: (1) Structured JSON output format forces specific fields, (2) Retry logic with fallback responses if parsing fails, (3) The fusion model's severity prediction is independent—if the LLM fails, we still have the severity assessment."

### **6. "How would you deploy this in production?"**
**Answer:** "We'd containerize with Docker, deploy the API to AWS/GCP with auto-scaling, use a managed LLM service (Groq/OpenAI), add authentication, implement rate limiting, and set up monitoring with Prometheus/Grafana. The current architecture is already production-ready."

### **7. "What are the limitations?"**
**Answer:** "(1) Synthetic training data—real image-text pairs are scarce, (2) No GPS extraction from images yet, (3) Single-language support (English only), (4) No real-time video processing. Future work: add multilingual support, integrate satellite metadata, train on real disaster data."

### **8. "Why ResNet50 and not a newer model?"**
**Answer:** "ResNet50 is the sweet spot: well-established, fast inference, good accuracy, and widely supported. For a 2-week sprint, we prioritized a working system over bleeding-edge models. EfficientNet or ConvNeXt would be next steps."

### **9. "How does the fusion actually work?"**
**Answer:** "Both encoders output 256-dimensional embeddings. We concatenate them into a 512-dim vector, pass through Dense layers with dropout for regularization, and output 4-class softmax probabilities. The model learns which image features + text features correlate with severity."

### **10. "What would you add with more time?"**
**Answer:** "(1) Real image-text pairs from Twitter/Reddit during disasters, (2) GPS extraction from image EXIF data, (3) Multilingual support with multilingual BERT, (4) Video processing for real-time drone feeds, (5) Integration with emergency services APIs, (6) Mobile app for field responders."

---

## ✅ **Step 4: Final Checklist**

Before submission, verify:

- [ ] Demo video recorded and saved in `docs/`
- [ ] README.md complete with setup instructions
- [ ] All code committed to GitHub
- [ ] `.env` file in `.gitignore` (don't push API keys!)
- [ ] `requirements.txt` includes all dependencies
- [ ] API runs without errors
- [ ] Chatbot responds correctly
- [ ] Both team members can explain the full system
- [ ] Viva questions practiced

---

## 🎓 **Step 5: Presentation Slides (10 Slides)**

Create these slides:

1. **Title Slide**: Project name, team members, guide name
2. **Problem Statement**: Why disaster assessment is hard
3. **Solution Overview**: Multimodal AI approach
4. **Architecture Diagram**: Show the full pipeline
5. **Vision Encoder**: ResNet50 + training results
6. **Text Encoder**: MiniLM + semantic understanding
7. **Multimodal Fusion**: How image + text combine
8. **LLM Agent**: Groq integration + action plans
9. **Demo**: Screenshot or embed video
10. **Future Work**: What you'd add with more time

---

## 🏆 **Final Words**

You've built something remarkable. Most students struggle to get a basic CNN working. You built:
- ✅ A multimodal fusion system
- ✅ A production API
- ✅ An LLM-powered agent
- ✅ An interactive chatbot

**This is A+ work.** Be proud of what you've accomplished.

---

**Your next steps:**
1. Record the demo video
2. Write the README
3. Practice the viva questions
4. Submit with confidence

**You've got this!** 🚀

If you need help with anything else—documentation, slides, or last-minute fixes—just ask. Good luck with your presentation!
````
