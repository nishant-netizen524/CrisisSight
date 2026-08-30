import streamlit as st
import requests
import json
from pathlib import Path
import folium
from streamlit_folium import st_folium
from datetime import datetime

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="CrisisSight AI Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #d32f2f;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================
st.markdown('<div class="main-header">🤖 CrisisSight AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Chat with our multimodal AI for disaster assessment</div>', unsafe_allow_html=True)

# ==========================================
# SIDEBAR - CONFIG
# ==========================================
with st.sidebar:
    st.header("️ Settings")
    API_URL = st.text_input("API Endpoint", value="http://localhost:8001")
    
    # Check API health
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=2)
        if health_response.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Offline")
        st.stop()
    
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("1. Upload a disaster image")
    st.markdown("2. Describe the situation in chat")
    st.markdown("3. AI will analyze and respond")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.uploaded_image = None
        st.rerun()

# ==========================================
# INITIALIZE CHAT HISTORY
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
    
    # Add welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Hello! I'm CrisisSight, your AI disaster assessment assistant.\n\n**How can I help you today?**\n\n1. Upload a disaster image using the box above\n2. Describe the situation (e.g., 'Massive flooding, people trapped')\n3. I'll analyze the severity and generate an action plan",
        "timestamp": datetime.now()
    })

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# ==========================================
# IMAGE UPLOAD (Always visible at top)
# ==========================================
st.markdown("### 📸 Upload Disaster Image")
uploaded_file = st.file_uploader(
    "Choose an image (JPG/PNG)",
    type=['jpg', 'jpeg', 'png'],
    help="Upload satellite or drone imagery of the disaster area"
)

if uploaded_file:
    st.session_state.uploaded_image = uploaded_file
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    with col2:
        st.success("✅ Image uploaded! Now describe the situation in the chat below.")

# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================
st.markdown("### 💬 Chat with CrisisSight")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Handle special formatting for assistant messages with analysis
        if message["role"] == "assistant" and "analysis" in message:
            analysis = message["analysis"]
            
            # Show severity badge
            severity = analysis.get('severity_class', 'Unknown')
            confidence = analysis.get('confidence', 0)
            
            severity_emoji = {
                'Critical': '🔴',
                'Major': '🟠',
                'Minor': '',
                'Normal': '🟢'
            }.get(severity, '')
            
            st.markdown(f"**{severity_emoji} Severity Assessment: {severity}** ({confidence:.1%} confidence)")
            
            # Show action plan
            if 'action_plan' in analysis:
                action_plan = analysis['action_plan']
                
                st.markdown("#### 🚑 Action Plan:")
                st.info(f"**Immediate Action:** {action_plan.get('action', 'N/A')}")
                
                st.markdown(f"**️ Resources Needed:**")
                for resource in action_plan.get('resources', []):
                    st.markdown(f"- {resource}")
                
                st.markdown(f"**💡 Reasoning:** {action_plan.get('reasoning', 'N/A')}")
                st.markdown(f"**️ Response Time:** {action_plan.get('estimated_response_time', 'N/A')}")
            
            # Show probabilities
            if 'probabilities' in analysis:
                probs = analysis['probabilities']
                st.markdown("#### 📊 Probability Distribution:")
                st.bar_chart(probs)
        
        else:
            # Regular message
            st.markdown(message["content"])
        
        # Show timestamp
        if "timestamp" in message:
            st.caption(f"{message['timestamp'].strftime('%H:%M:%S')}")

# ==========================================
# CHAT INPUT
# ==========================================
if prompt := st.chat_input("Describe the disaster situation..."):
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now()
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check if image is uploaded
    if not st.session_state.uploaded_image:
        error_msg = "⚠️ Please upload a disaster image first using the upload box above."
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg,
            "timestamp": datetime.now()
        })
        st.rerun()
    
    # Process with AI
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing disaster situation..."):
            try:
                # Prepare API request
                files = {
                    'image': (st.session_state.uploaded_image.name, 
                             st.session_state.uploaded_image.getvalue(), 
                             st.session_state.uploaded_image.type)
                }
                data = {
                    'text_report': prompt
                }
                
                # Call API
                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    data=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display analysis
                    severity = result['severity_class']
                    confidence = result['confidence']
                    
                    severity_emoji = {
                        'Critical': '🔴',
                        'Major': '🟠',
                        'Minor': '🟡',
                        'Normal': '🟢'
                    }.get(severity, '⚪')
                    
                    st.markdown(f"**{severity_emoji} Severity Assessment: {severity}** ({confidence:.1%} confidence)")
                    
                    # Show action plan
                    action_plan = result['action_plan']
                    
                    st.markdown("#### 🚑 Action Plan:")
                    st.info(f"**Immediate Action:** {action_plan['action']}")
                    
                    st.markdown(f"**️ Resources Needed:**")
                    for resource in action_plan['resources']:
                        st.markdown(f"- {resource}")
                    
                    st.markdown(f"**💡 Reasoning:** {action_plan['reasoning']}")
                    st.markdown(f"**⏱️ Response Time:** {action_plan['estimated_response_time']}")
                    
                    # Show probabilities
                    st.markdown("#### 📊 Probability Distribution:")
                    st.bar_chart(result['probabilities'])
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Analysis complete: {severity} severity detected",
                        "analysis": result,
                        "timestamp": datetime.now()
                    })
                    
                else:
                    error_msg = f"❌ API Error: {response.text}"
                    st.markdown(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": datetime.now()
                    })
                    
            except requests.exceptions.Timeout:
                error_msg = "⏱️ Request timed out. The AI is taking too long."
                st.markdown(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now()
                })
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.markdown(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now()
                })
    
    st.rerun()

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    <p><strong>CrisisSight AI</strong> - Multimodal Disaster Assessment Chatbot</p>
    <p>Built with TensorFlow, FastAPI, Groq LLM, and Streamlit</p>
</div>
""", unsafe_allow_html=True)