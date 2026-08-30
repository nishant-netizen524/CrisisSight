import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class CrisisAgent:
    """
    AI Disaster Response Coordinator.
    Uses Groq's Llama-3-70B to generate structured action plans
    based on severity assessments and ground reports.
    """
    
    SYSTEM_PROMPT = """You are CrisisSight, an expert AI disaster response coordinator.
Your job is to analyze severity assessments and generate actionable response plans.

You MUST respond with ONLY valid JSON in this exact format:
{
    "severity": "<low|medium|high|critical>",
    "action": "<one-sentence immediate action>",
    "resources": ["<resource1>", "<resource2>", "<resource3>"],
    "reasoning": "<brief explanation of why this action is needed>",
    "estimated_response_time": "<time estimate>"
}

Rules:
- Be specific and practical
- Resources should be real (ambulance, fire truck, rescue boat, medical team, etc.)
- Reasoning should connect the severity with the ground report
- Keep responses concise but actionable
- Output ONLY the JSON, no extra text or markdown"""
    
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "groq")
        self.api_key = os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "❌ GROQ_API_KEY not found in .env file. "
                "Get a free key at https://console.groq.com"
            )
        
        # Initialize Groq client
        from groq import Groq
        self.client = Groq(api_key=self.api_key)
        print("✅ CrisisAgent initialized (Groq qwen/qwen3.8-27b)")
    
    def _parse_json_response(self, text: str) -> dict:
        """Extract JSON from LLM response, handling markdown wrappers"""
        text = text.strip()
        
        # Remove markdown code blocks if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        return json.loads(text)
    
    def generate_report(
        self,
        severity: str,
        image_context: str,
        text_report: str,
        max_retries: int = 2
    ) -> dict:
        """
        Generate a disaster response action plan.
        
        Args:
            severity: Predicted severity (Normal/Minor/Major/Critical)
            image_context: Description of what the vision model detected
            text_report: Ground report or user description
            max_retries: Number of retry attempts on failure
        
        Returns:
            dict: Structured action plan
        """
        user_prompt = f"""DISASTER ASSESSMENT:
- Predicted Severity: {severity}
- Visual Analysis: {image_context}
- Ground Report: {text_report}

Generate an actionable response plan in JSON format."""
        
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model="qwen/qwen3.8-27b",
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500,
                    response_format={"type": "json_object"}  # Force JSON output
                )
                
                result_text = response.choices[0].message.content
                return self._parse_json_response(result_text) #type: ignore
                
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON parse error (attempt {attempt+1}): {e}")
                if attempt == max_retries:
                    return self._fallback_response(severity, str(e))
                time.sleep(1)
                
            except Exception as e:
                error_str = str(e).lower()
                # Handle rate limits
                if "rate limit" in error_str or "429" in error_str:
                    print(f"⚠️ Rate limit hit, waiting 5s... (attempt {attempt+1})")
                    time.sleep(5)
                    continue
                
                print(f"⚠️ LLM error (attempt {attempt+1}): {e}")
                if attempt == max_retries:
                    return self._fallback_response(severity, str(e))
                time.sleep(2)
        
        return self._fallback_response(severity, "Max retries exceeded")
    
    def _fallback_response(self, severity: str, error_msg: str) -> dict:
        """Return a safe fallback if LLM fails"""
        severity_lower = severity.lower()
        
        if severity_lower == "critical":
            action = "Immediate emergency response required"
            resources = ["Ambulance", "Fire rescue team", "Search and rescue unit", "Medical supplies"]
            time_est = "Immediate (0-15 minutes)"
        elif severity_lower == "major":
            action = "Dispatch emergency response team"
            resources = ["Emergency services", "Medical team", "Relief supplies"]
            time_est = "30-60 minutes"
        elif severity_lower == "minor":
            action = "Monitor situation and prepare resources"
            resources = ["Local response team", "Basic supplies"]
            time_est = "1-2 hours"
        else:
            action = "No immediate action required"
            resources = ["Routine monitoring"]
            time_est = "N/A"
        
        return {
            "severity": severity_lower,
            "action": action,
            "resources": resources,
            "reasoning": f"AI agent fallback due to: {error_msg}",
            "estimated_response_time": time_est
        }


# ==========================================
# TEST THE AGENT STANDALONE
# ==========================================
if __name__ == "__main__":
    print("🧪 Testing CrisisAgent standalone...")
    
    agent = CrisisAgent()
    
    test_cases = [
        {
            "severity": "Critical",
            "image_context": "Collapsed building with visible structural damage",
            "text_report": "Multiple people trapped, urgent rescue needed"
        },
        {
            "severity": "Normal",
            "image_context": "Clear road with normal traffic flow",
            "text_report": "Routine patrol completed, no incidents"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"📝 Test Case {i}: {test['severity']} severity")
        print(f"{'='*60}")
        
        result = agent.generate_report(**test)  #type: ignore
        print(json.dumps(result, indent=2))
    
    print("\n✅ CrisisAgent test complete!")