import os
import json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
token = os.getenv("HF_API_TOKEN")
model_id = "Qwen/Qwen2.5-7B-Instruct"

print(f"Testing Model: {model_id}")
print(f"Token found: {token[:10]}..." if token else "Token not found")

client = InferenceClient(model=model_id, token=token)

try:
    # Small test query
    res = client.chat_completion(
        messages=[{"role": "user", "content": "Tell me one short fact about banking."}], 
        max_tokens=50
    )
    print("\nSuccess!")
    print("Response:", res.choices[0].message.content)
except Exception as e:
    print("\nError:", e)
