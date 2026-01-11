import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq

# 1. Load Environment Variables
load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

print(f"🔑 Key found: {api_key[:5]}...")

# 2. Initialize with the NEW Model ID
# OLD: "llama3-70b-8192" (Decommissioned)
# NEW: "llama-3.3-70b-versatile" (Active)
try:
    print("🤖 Testing connection to Groq Llama 3.3...")
    llm = Groq(model="llama-3.3-70b-versatile", api_key=api_key)
    
    response = llm.complete("Hello! Are you working?")
    print(f"🎉 Success! The model replied: {response}")
except Exception as e:
    print(f"❌ Failed: {e}")