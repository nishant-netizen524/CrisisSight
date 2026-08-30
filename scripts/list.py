import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("📋 Available models on your Groq account:\n")
models = client.models.list()
for model in models.data:
    print(f"  • {model.id}")