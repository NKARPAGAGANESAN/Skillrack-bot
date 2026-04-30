import os
import re
import time
import requests
import certifi
from dotenv import load_dotenv

# Fix SSL certificates on Windows for Google SDK
os.environ['SSL_CERT_FILE'] = certifi.where()

load_dotenv()

# ================= CONFIG =================
# Gemini API Keys Pool (Rotating)
GEMINI_KEYS = [
    "AIzaSyDOappr8f6GuRqA869tTBGp7-lDrP9teG4",
    "AIzaSyBPKKGiEjaoWhAZxeyeGJxdHR-qhe654jA",
    "AIzaSyDWFmBY5qKqDGiz1FVv9tpm63p-W-2GeW4",
    "AIzaSyDScHcvsrRgLeCTXl2umsH_TzbmikOfEB",
    "AIzaSyD2CBOLEYTrW0D-KmSeyOqKEMLbHxxx76o",
    "AIzaSyBy6ApxIZLW69V86wlNIN2JL0ROZ_WlVF4",
    "AIzaSyDXpRyGs4Q78MqtdX7P_3jT_KvpV2GVTLY",
    "AIzaSyDXFjWW8_7rvdmhAb0JrmjCEGBuuQeJCi0",
    "AIzaSyD0WTEj6mRRx9OlfNnzOGkZHa7puB-z_KU",
    "AIzaSyDScHcvsrRgLeCTXl2umsH_TzbmikOfEB",
    "AQ.Ab8RN6I0FMcoslGBW0UseaSJLiTBXC2J4GUBRs4kCX0TENAhDg"
]

# Add key from .env if present
_env_key = os.getenv("GOOGLE_API_KEY")
if _env_key and _env_key not in GEMINI_KEYS:
    GEMINI_KEYS.insert(0, _env_key)

# Model Definitions - STRICTLY Gemini 2.5 Flash and Flash-Lite
MODELS = [
    {"name": "Gemini 2.5 Flash", "provider": "google", "id": "gemini-2.5-flash"},
    {"name": "Gemini 2.5 Flash-Lite", "provider": "google", "id": "gemini-2.5-flash-lite"},
]

# State for Key Rotation
_key_index = 0
_key_usage = {key: [] for key in GEMINI_KEYS} # Track timestamps of requests per key

def _get_next_valid_key():
    global _key_index
    start_idx = _key_index
    
    while True:
        key = GEMINI_KEYS[_key_index]
        now = time.time()
        # Clean up timestamps older than 60 seconds
        if key not in _key_usage:
            _key_usage[key] = []
        _key_usage[key] = [t for t in _key_usage[key] if now - t < 60]
        
        # If this key has used fewer than 10 requests in the last minute, use it
        if len(_key_usage[key]) < 10:
            _key_usage[key].append(now)
            # Advance index for next time to ensure distribution
            current_key = key
            _key_index = (_key_index + 1) % len(GEMINI_KEYS)
            return current_key
            
        # Move to next key
        _key_index = (_key_index + 1) % len(GEMINI_KEYS)
        
        # If we've checked all keys and they are all busy, wait a bit
        if _key_index == start_idx:
            print("[*] All Gemini keys are at the 10 RPM limit. Waiting 10s...")
            time.sleep(10)

def _build_prompt(problem_text: str, language: str = "") -> str:
    lang = (language or "Python3").strip()
    is_java = "java" in lang.lower()
    
    java_constraint = ""
    if is_java:
        java_constraint = """
### JAVA CRITICAL RULES:
- YOU MUST START WITH: `import java.util.*;`
- YOU MUST DEFINE: `public class Hello { ... }`
- THE MAIN METHOD MUST BE INSIDE `public class Hello`.
- NEVER output just the method body.
"""

    return f"""TASK: Write a COMPLETE, RUNNABLE {lang} program.

### MANDATORY REQUIREMENTS:
1. NO SNIPPETS. You MUST output the ENTIRE class from scratch.
2. IMPORTANT: The problem description below may contain fragments of starter code (e.g., "1- import", "public class"). IGNORE THESE COMPLETELY. DO NOT try to complete them. Start from a blank slate.
3. {java_constraint}
4. JAVA INPUT: For reading matrices or lists of numbers, ALWAYS use `sc.nextInt()` in a loop. DO NOT use `sc.nextLine().split()` as it is fragile.
5. OUTPUT SPACES: Ensure every number/character in the output is separated by a space as per the rules.
6. NO EXPLANATIONS. NO MARKDOWN. NO COMMENTS. Just the raw code.
7. MATCH EXAMPLES EXACTLY.

### PROBLEM DESCRIPTION (WARNING: MAY CONTAIN FRAGMENTS TO IGNORE):
{problem_text}

### OUTPUT START:
(Write the FULL {lang} code here)
"""

# ================= PROVIDERS =================

def _call_google(model_id: str, prompt: str) -> str:
    from google import genai
    from google.genai import types
    
    # Get a key that hasn't exceeded 10 RPM
    api_key = _get_next_valid_key()
    client = genai.Client(api_key=api_key)
    
    config = types.GenerateContentConfig(
        system_instruction="You are a competitive programming expert. Output ONLY raw code. No markdown. No explanations.",
        temperature=0.1,
    )
    
    errors = []
    # Try the exact model ID, then common suffixed variants
    variations = [
        model_id,
        f"{model_id}-latest",
    ]
    
    # Allow rotating through ALL keys in the pool if one fails
    for retry_attempt in range(len(GEMINI_KEYS)):
        for mid in variations:
            try:
                response = client.models.generate_content(model=mid, contents=prompt, config=config)
                return response.text
            except Exception as e:
                err = str(e)
                if "429" in err or "quota" in err.lower():
                    print(f"[!] Key {api_key[:10]}... hit rate limit. Rotating to key #{retry_attempt+1}...")
                    api_key = _get_next_valid_key()
                    client = genai.Client(api_key=api_key)
                    break # Break inner loop to try NEW key
                elif "API key not valid" in err or "400" in err or "403" in err:
                    print(f"[!] Key {api_key[:10]}... failed ({err[:50]}). Trying next key...")
                    api_key = _get_next_valid_key()
                    client = genai.Client(api_key=api_key)
                    break # Break inner loop
                errors.append(err)
                continue
    
    raise RuntimeError(f"Gemini failed on all {len(GEMINI_KEYS)} keys: {errors[0] if errors else 'Unknown error'}")

def _call_groq(model_id: str, prompt: str) -> str:
    # Groq is disabled as per user request to use ONLY Gemini
    return ""

# ================= MAIN API =================

def _simple_fallback(problem_text: str, language: str = "") -> str | None:
    """Very small heuristic for trivial problems (e.g., print numbers 1..N)."""
    lowered = problem_text.lower()
    if "print the numbers from 1 to n" in lowered or "print numbers from 1 to n" in lowered:
        if "java" in language.lower():
            return (
                "import java.util.*;\n"
                "public class Hello {\n"
                "    public static void main(String[] args) {\n"
                "        Scanner sc = new Scanner(System.in);\n"
                "        int n = sc.nextInt();\n"
                "        StringBuilder sb = new StringBuilder();\n"
                "        for (int i = 1; i <= n; i++) sb.append(i);\n"
                "        System.out.print(sb.toString());\n"
                "    }\n"
                "}\n"
            )
        # default to Python
        return "n = int(input()); print(''.join(str(i) for i in range(1, n+1)))"
    return None

def generate_solution(problem_text: str, language: str = "", **kwargs) -> str:
    prompt = _build_prompt(problem_text, language)
    
    for model_info in MODELS:
        name = model_info["name"]
        provider = model_info["provider"]
        model_id = model_info["id"]
        
        try:
            print(f"[*] Trying {name}...")
            if provider == "google":
                response = _call_google(model_id, prompt)
            else:
                response = _call_groq(model_id, prompt)
            
            if response and len(response.strip()) > 10:
                # Extract code, handling possible markdown fences
                code_text = response.strip()
                code_match = re.search(r"```[a-zA-Z]*\n?([\s\S]*?)```", response)
                if code_match:
                    code_text = code_match.group(1).strip()
                
                # Post-process for Java: ensure proper class Hello and imports
                if "java" in language.lower():
                    # 1. Strip markdown fences if still present
                    code_text = re.sub(r"```[a-zA-Z]*\n?", "", code_text).replace("```", "")
                    
                    # 2. Ensure "public class Hello"
                    # If it has a class but it's not Hello, rename it
                    if "class " in code_text and "class Hello" not in code_text:
                        code_text = re.sub(r"class\s+\w+", "class Hello", code_text)
                    
                    # 3. If "class Hello" is missing entirely
                    if "class Hello" not in code_text:
                        print("[!] Java class Hello missing. Wrapping automatically...")
                        # Check if it already has a main method
                        if "public static void main" in code_text:
                            code_text = "public class Hello {\n" + code_text + "\n}"
                        else:
                            # Wrap everything in class + main
                            code_text = "public class Hello {\n    public static void main(String[] args) {\n" + code_text + "\n    }\n}"
                    
                    # 4. Ensure import java.util.*; is present at top
                    if "import java.util" not in code_text:
                        code_text = "import java.util.*;\n" + code_text
                        
                return code_text.strip()
                
        except Exception as e:
            err_str = str(e)
            # Rate‑limit handling – respect retry‑after when present
            if "429" in err_str:
                # Extract possible "try again in XmYs" hint
                import re as _re
                m = _re.search(r"try again in (\d+)m(\d+\.\d+)s", err_str)
                if m:
                    wait_sec = int(m.group(1)) * 60 + float(m.group(2))
                else:
                    # fallback wait time if no hint is present
                    wait_sec = 30
                print(f"[!] {name} RATE LIMIT REACHED (429). Waiting {wait_sec:.1f}s...")
                time.sleep(wait_sec)
                # retry the same model once more after the pause
                try:
                    if provider == "google":
                        response = _call_google(model_id, prompt)
                    else:
                        response = _call_groq(model_id, prompt)
                    if response:
                        return response.strip()
                except Exception:
                    pass
            else:
                print(f"[!] {name} failed: {err_str[:120]}...")
                continue

            
    # All remote models failed – try simple heuristic fallback
    print("[!] All remote models failed.")
    fallback = _simple_fallback(problem_text, language)
    if fallback:
        print("[+] Heuristic fallback generated a solution.")
        return fallback
    raise RuntimeError("All models failed.")

def generate_repair(problem_text: str, code: str, error_msg: str, language: str = "") -> str:
    lang = (language or "the same language").strip()
    repair_prompt = (
        f"The following {lang} code failed with this error:\n\n"
        f"Error/Observation:\n{error_msg}\n\n"
        f"Original code:\n{code}\n\n"
        f"Problem description:\n{problem_text}\n\n"
        f"Fix the code. Output ONLY the complete corrected {lang} program, no explanations."
    )
    return generate_solution(repair_prompt, language=language)