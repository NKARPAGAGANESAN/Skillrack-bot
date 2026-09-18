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
# Only verified active keys are retained in default pool
GEMINI_KEYS = [
    "AQ.Ab8RN6LTzMQtKKhTkAne1NLcxqqluow6oxJ22gnAmuH0hJjZsA",
    "AIzaSyDm2tlqjiZzePcRb17JUerIBY7F_ctvDaw"
]

# Add primary key from .env if present
_env_key = os.getenv("GOOGLE_API_KEY")
if _env_key and _env_key not in GEMINI_KEYS:
    GEMINI_KEYS.insert(0, _env_key)

# Add any additional comma-separated keys from .env
_extra_keys = os.getenv("GEMINI_API_KEYS", "")
for k in _extra_keys.split(","):
    k = k.strip()
    if k and k not in GEMINI_KEYS:
        GEMINI_KEYS.append(k)

# Model Definitions - Gemini 2.5 Flash and Groq models provide reliable, instant generation
MODELS = [
    {"name": "Gemini 2.5 Flash", "provider": "google", "id": "gemini-2.5-flash"},
    {"name": "Groq GPT-OSS (120B)", "provider": "groq", "id": "openai/gpt-oss-120b"},
    {"name": "Groq Qwen 3.8 (27B)", "provider": "groq", "id": "qwen/qwen3.8-27b"},
    {"name": "Gemini 3.6 Flash", "provider": "google", "id": "gemini-3.6-flash"},
]

# State for Key Rotation & Health
_key_index = 0
_key_usage = {key: [] for key in GEMINI_KEYS} # Track timestamps of requests per key
_blacklisted_keys = set() # Automatically blacklist invalid (400) or banned (403) keys

def _get_next_valid_key():
    global _key_index
    available_keys = [k for k in GEMINI_KEYS if k not in _blacklisted_keys]
    if not available_keys:
        raise RuntimeError("No valid Gemini API keys available. All keys exhausted or blacklisted.")
    
    start_idx = _key_index % len(available_keys)
    while True:
        key = available_keys[_key_index % len(available_keys)]
        now = time.time()
        
        # Advance index for the next request attempt
        _key_index = (_key_index + 1) % len(available_keys)
        
        # Clean up timestamps older than 60 seconds
        if key not in _key_usage:
            _key_usage[key] = []
        _key_usage[key] = [t for t in _key_usage[key] if now - t < 60]
        
        # Check RPM limit (12 RPM per key gives a safe buffer under the 15 RPM cap)
        if len(_key_usage[key]) < 12:
            _key_usage[key].append(now)
            return key
            
        if (_key_index % len(available_keys)) == start_idx:
            print("[*] All healthy Gemini keys are near their RPM limit. Waiting 3s...")
            time.sleep(3)

def _build_prompt(problem_text: str, language: str = "") -> str:
    lang = (language or "Python3").strip()
    is_java = "java" in lang.lower()
    is_c = bool(re.search(r"\bC\b|^\s*c\s*\(", lang, re.IGNORECASE)) and "c++" not in lang.lower() and "cpp" not in lang.lower()
    
    lang_constraint = ""
    if is_java:
        lang_constraint = """
### JAVA CRITICAL RULES:
- YOU MUST START WITH: `import java.util.*;`
- YOU MUST DEFINE: `public class Hello { ... }`
- THE MAIN METHOD MUST BE INSIDE `public class Hello`.
- Read tokens safely with `Scanner sc = new Scanner(System.in);`.
- NEVER output just the method body.
"""
    elif is_c:
        lang_constraint = """
### C CRITICAL RULES:
- YOU MUST INCLUDE: `#include <stdio.h>`, `#include <stdlib.h>`, `#include <string.h>`, `#include <math.h>`, `#include <ctype.h>`
- YOU MUST DEFINE: `int main() { ... return 0; }`
- BUFFER SIZE: ALWAYS use large buffers for string/line input: `char s[100005];` (never small buffers).
- STRING INPUT WITH SPACES: When reading a string/line that may contain spaces, ALWAYS use `scanf(" %[^\n]", s);` or `fgets(s, sizeof(s), stdin); s[strcspn(s, "\\r\\n")] = 0;`.
- NEVER use `gets()`.
- NEVER output just snippets or functions without main().
"""
    else:
        lang_constraint = """
### PYTHON CRITICAL RULES:
- Read input cleanly: for space-separated or multi-line tokens, use `sys.stdin.read().split()` or `sys.stdin.read().splitlines()`.
- Format output EXACTLY as required. Do not print extra trailing spaces.
- Pay attention to 1-based vs 0-based indexing as specified in problem examples.
"""

    return f"""TASK: Write a COMPLETE, RUNNABLE {lang} program for the problem below.
    
### LOGIC RULES:
- EXAMPLES: Pay closest attention to the EXAMPLE INPUT and EXAMPLE OUTPUT. Your code must match the example outputs EXACTLY.
- TEXT SOURCES: You will see 'WEB TEXT' and 'OCR TEXT'. Use WEB TEXT as the primary source, but verify examples from OCR.
- FIX OCR TYPOS: Correct typos like 'MXN' to 'M x N' using problem context.
- MATRIX ROTATION: CLOCKWISE: res[j][m-1-i]=mat[i][j] | ANTI-CLOCKWISE: res[n-1-j][i]=mat[i][j]
- PATTERNS: For complex grids, pre-fill a 2D matrix before printing.
- FORMATTING: Use exactly one space (' ') between values unless problem requires otherwise.

### STRICTURE RULES:
1. NO SNIPPETS. Output the ENTIRE class/program.
2. {lang_constraint}
3. NO EXPLANATIONS. Just raw code.

### PROBLEM DESCRIPTION:
{problem_text}

### OUTPUT START:
"""

# ================= PROVIDERS =================

def _call_google(model_id: str, prompt: str) -> str:
    from google import genai
    from google.genai import types
    
    config = types.GenerateContentConfig(
        system_instruction="You are a competitive programming expert. Output ONLY raw code. No markdown. No explanations.",
        temperature=0.1,
    )
    
    errors = []
    
    # Preferred model variations to try in order on each key
    candidate_variations = [
        model_id,
        "gemini-2.5-flash",
        "gemini-3.6-flash",
    ]
    variations = []
    for v in candidate_variations:
        if v and v not in variations:
            variations.append(v)
    
    available_keys = [k for k in GEMINI_KEYS if k not in _blacklisted_keys]
    max_attempts = max(len(available_keys) * 2, 3)
    
    for attempt in range(max_attempts):
        try:
            api_key = _get_next_valid_key()
        except RuntimeError as e:
            raise RuntimeError(f"All Gemini keys exhausted or blacklisted: {e}")
            
        try:
            client = genai.Client(api_key=api_key)
        except Exception as e:
            errors.append(f"Client init error on key {api_key[:8]}: {e}")
            continue

        for mid in variations:
            try:
                response = client.models.generate_content(
                    model=mid, 
                    contents=prompt, 
                    config=config
                )
                if response and response.text:
                    return response.text
            except Exception as e:
                err = str(e)
                
                # Check for permanent invalid / permission denied (dead project)
                if "400" in err or "403" in err or "invalid" in err.lower() or "denied" in err.lower():
                    print(f"[!] Key {api_key[:10]}... is invalid or banned ({err[:50]}). Blacklisting...")
                    _blacklisted_keys.add(api_key)
                    errors.append(f"Blacklisted key {api_key[:8]}")
                    break # Break out of variations loop to try next key
                
                # Check for rate limit
                if "429" in err or "quota" in err.lower():
                    errors.append(f"429 on {mid} (key {api_key[:8]})")
                    print(f"[!] Key {api_key[:10]}... hit rate limit on {mid}. Trying next variation/key...")
                    continue # Try next variation on this key (e.g. 2.5-flash) before moving to next key!
                
                # Model 404 / unsupported on this key -> try next variation on SAME key
                if "404" in err or "not found" in err.lower():
                    errors.append(f"404 on {mid} (key {api_key[:8]})")
                    continue # Try next variation!
                
                errors.append(err[:80])
                continue

    error_summary = ", ".join(list(dict.fromkeys(errors))[:5])
    raise RuntimeError(f"Gemini failed across healthy keys. Errors: {error_summary or 'No response'}")

def _call_groq(model_id: str, prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return ""
    
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model_id,
            "messages": [
                {
                    "role": "system", 
                    "content": (
                        "You are a competitive programming expert. "
                        "STRICT LOGIC RULES:\n"
                        "1. EXAMPLES: Pay closest attention to the EXAMPLE INPUT and EXAMPLE OUTPUT. Your code must match them EXACTLY.\n"
                        "2. C: ALWAYS include <stdio.h>, <stdlib.h>, <string.h>, <math.h>, <ctype.h>. Use 'int main() { ... return 0; }'.\n"
                        "   BUFFER SIZE: Use char s[100005] for strings. Read string lines with scanf(\" %[^\\n]\", s) or fgets.\n"
                        "3. PYTHON: Use sys.stdin.read().split() or sys.stdin.read().splitlines() for multi-token / multi-line inputs.\n"
                        "4. JAVA: ALWAYS 'public class Hello'. Use 'Scanner sc = new Scanner(System.in);'.\n"
                        "5. FORMATTING: One space (' ') between values unless specified otherwise. No trailing spaces.\n"
                        "6. NO markdown, no explanations. Output ONLY raw code."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1
        }
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        print(f"[!] Groq error: {e}")
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

def generate_solution(problem_text: str, language: str = "", smart_priority: bool = False, **kwargs) -> str:
    prompt = _build_prompt(problem_text, language)
    
    # Filter models
    current_models = MODELS[:]
    
    if smart_priority:
        # Prioritize frontier reasoning models for repairs
        current_models.sort(key=lambda x: "120b" in x["id"] or "2.5" in x["id"], reverse=True)
    
    for model_info in current_models:
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
                        
                is_c = bool(re.search(r"\bC\b|^\s*c\s*\(", language, re.IGNORECASE)) and "c++" not in language.lower() and "cpp" not in language.lower()
                if is_c:
                    code_text = re.sub(r"```[a-zA-Z]*\n?", "", code_text).replace("```", "").strip()
                    if "<stdio.h>" not in code_text:
                        code_text = "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <math.h>\n#include <ctype.h>\n\n" + code_text

                    # Guarantee that the program is enclosed in int main()
                    if "main(" not in code_text and "main (" not in code_text:
                        print("[!] C main() function missing. Wrapping body in int main() automatically...")
                        headers = "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <math.h>\n#include <ctype.h>\n\n"
                        body = re.sub(r"#include\s*<[^>]+>\s*", "", code_text).strip()
                        code_text = headers + "int main() {\n" + body
                        if not code_text.rstrip().endswith("}"):
                            code_text += "\n    return 0;\n}\n"

                return code_text.strip()
                
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                print(f"[!] {name} hit rate limit (429). Instantly falling through to next model...")
                continue
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
    # Always use the smarter model first for repairs
    return generate_solution(repair_prompt, language=language, smart_priority=True)