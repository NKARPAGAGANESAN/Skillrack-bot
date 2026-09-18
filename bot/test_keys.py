import os
import time
from dotenv import load_dotenv
from google import genai
from llm_adapter import GEMINI_KEYS

load_dotenv()

def test_keys():
    print(f"Testing {len(GEMINI_KEYS)} keys...")
    for i, key in enumerate(GEMINI_KEYS):
        print(f"[{i+1}/{len(GEMINI_KEYS)}] Testing key {key[:10]}...")
        try:
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                model="gemini-3.6-flash",
                contents="Say 'OK'",
            )
            print(f"  [+] Success: {response.text.strip()}")
        except Exception as e:
            print(f"  [!] Failed: {str(e)[:100]}")
        time.sleep(1) # Be nice to the API

if __name__ == "__main__":
    test_keys()
