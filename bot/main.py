import ast
import difflib
import os
import re
import subprocess
import sys
import textwrap
import time
import cv2
import numpy as np
try:
    import pytesseract
    from PIL import Image
    # Updated path for Tesseract on D: drive
    pytesseract.pytesseract.tesseract_cmd = r''#load pytesseract directory here 
except ImportError:
    pytesseract = None

from dotenv import load_dotenv
load_dotenv()

# Global EasyOCR reader to avoid repeated initialization (prevents system hangs)
_EASYOCR_READER = None

from playwright.sync_api import TimeoutError, sync_playwright

from llm_adapter import generate_solution, generate_repair

# ================= CONFIG =================
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_TEXT_LEN = int(os.getenv("MAX_TEXT_LEN", "0"))
HEADLESS = os.getenv("HEADLESS", "0").strip().lower() in ("1", "true", "yes")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").strip().lower()
# Keep disabled by default; Ambiance often blocks paste operations.
MANUAL_PASTE_FALLBACK = os.getenv("MANUAL_PASTE_FALLBACK", "0").strip().lower() in ("1", "true", "yes")
TYPE_DELAY_MS = int(os.getenv("TYPE_DELAY_MS", "2"))
ENABLE_SAMPLE_CHECK = os.getenv("ENABLE_SAMPLE_CHECK", "1").strip().lower() in ("1", "true", "yes")
SAMPLE_CHECK_TIMEOUT_SEC = int(os.getenv("SAMPLE_CHECK_TIMEOUT_SEC", "3"))


# ================= HELPERS =================
def _normalize_code_indentation(code: str) -> str:
    """Aggressively fixes indentation and removes common LLM artifacts."""
    if not code: return ""
    
    # Remove markdown code fences if they survived extraction
    code = re.sub(r"```[a-zA-Z]*\n?", "", code).replace("```", "")
    
    # Normalize line endings and non-breaking spaces
    code = code.replace("\r", "").replace("\u00A0", " ")
    
    lines = code.split("\n")
    cleaned = []
    for ln in lines:
        # Remove any leading '>' characters from LLM blockquotes
        ln = re.sub(r"^\s*>\s*", "", ln)
        # Convert tabs to 4 spaces consistently
        ln = ln.replace("\t", "    ")
        cleaned.append(ln.rstrip())

    # Remove global indentation if all lines are indented (common in LLM responses)
    non_blank = [l for l in cleaned if l.strip()]
    if non_blank:
        indents = [len(re.match(r"^( *)", l).group(1)) for l in non_blank]
        min_indent = min(indents)
        if min_indent > 0:
            cleaned = [l[min_indent:] if len(l) >= min_indent else l for l in cleaned]

    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
        
    return "\n".join(cleaned)


# ================= BROWSER =================
def extract_problem_text(page):
    try:
        # Disabled scrolling to prevent "jumping" issues
        # page.evaluate("window.scrollTo(0, 300)")
        # time.sleep(0.1)
        # page.evaluate("window.scrollTo(0, 0)")

        # Extract full visible page text once (avoid duplicating tokens).
        txt = page.evaluate(
            """
            () => {
                const body = document.body;
                return body ? (body.innerText || "") : "";
            }
            """
        )
        # Remove excessive whitespace to keep prompt compact.
        txt = re.sub(r"[ \t]+", " ", txt or "")
        txt = re.sub(r"\\n{3,}", "\n\n", (txt or "").strip())
        if txt:
            return txt
        return page.inner_text("body")
    except Exception:
        return page.inner_text("body")


def extract_text_via_ocr(page) -> str:
    """Take a screenshot of the problem area and use Pytesseract to extract text."""
    if not pytesseract:
        print("[!] Pytesseract not installed. Falling back to innerText.")
        return ""
        
    try:
        # Skillrack problems are usually in this container
        container = page.locator("#codeprogramdiv, .program-container, #MainContainer").first
        if container.count() == 0:
            container = page.locator("body")
            
        print("[*] Taking screenshot for OCR...")
        screenshot_path = "problem_ocr.png"
        container.screenshot(path=screenshot_path)
        
        # KEEP FULL WIDTH (Examples are often on the right!)
        try:
            img = Image.open(screenshot_path)
            width, height = img.size
            # Only skip the top nav bar, keep 100% width
            img = img.crop((0, 120, width, height))
            img.save(screenshot_path)
        except Exception as e:
            print(f"[!] Cropping failed: {e}")

        print("[*] Running Pytesseract OCR...")
        text = pytesseract.image_to_string(Image.open(screenshot_path))
        
        # CLEANUP OCR
        text = re.sub(r"^\s*\d+\s*-\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"Valid Till:.*", "", text)
        text = re.sub(r"Home \| Reports.*", "", text)
        
        print(f"\n--- OCR TEXT CAPTURED ---\n{text[:500]}...\n------------------------\n")
        
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        return text
    except Exception as e:
        print(f"[!] OCR Error: {e}")
        return ""


def compact_problem_text(raw_text: str) -> str:
    txt = (raw_text or "").replace("\r", "")
    if not txt.strip():
        return txt

    # Only remove UI navigation chrome, preserving full problem, constraints, and examples
    noise_patterns = [
        r"\bHome\s+Reports\s+Profile\s+Help\s+Logout\b",
        r"\bui-button\b",
        r"Back To Challenges List",
        r"Valid Till:\s*[^\n]+",
        r"Run with a custom test case(\s*\(Input/Output\))?",
        r"Save\s+Run\s+Submit(?!ted)",
        r"Ambiance\s+[A-Za-z0-9_().+\s-]+\b",
    ]
    for pat in noise_patterns:
        txt = re.sub(pat, " ", txt, flags=re.IGNORECASE)

    # Normalize whitespace but preserve structure
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()

    # If Max Execution Time Limit is present at the end of the problem container, trim after it
    time_limit_idx = txt.lower().find("max execution time limit")
    if time_limit_idx != -1:
        end_idx = txt.find("\n", time_limit_idx)
        if end_idx != -1:
            txt = txt[:end_idx].strip()

    return txt.strip()


def get_selected_language(page) -> str:
    try:
        txt = page.evaluate(
            r"""
            () => {
                const norm = (s) => (s || "").trim();
                const labels = [...document.querySelectorAll(".ui-selectonemenu-label")]
                    .map((el) => norm(el.textContent))
                    .filter(Boolean);
                // Robust check for Java, Python, CPP, and standalone C
                const pick = labels.find((t) => /java|python|cpp|c\+\+|\bC\b|^\s*c\s*\(/i.test(t));
                if (pick) return pick;

                const sel = document.querySelector("select");
                if (sel) {
                    const selected = sel.options?.[sel.selectedIndex]?.text || sel.value || "";
                    if (/java|python|cpp|c\+\+|\bC\b|^\s*c\s*\(/i.test(selected)) return norm(selected);
                }
                return "";
            }
            """
        )
        return (txt or "").strip()
    except Exception:
        return ""


def get_editor_starter_code(page) -> str:
    """Read current code from editor (Ace, Monaco, CodeMirror, or textarea)."""
    get_code_js = """
        () => {
            // 1. Check Ace editors
            const aceEls = document.querySelectorAll('.ace_editor');
            for (const el of aceEls) {
                try {
                    if (window.ace && window.ace.edit) {
                        const val = window.ace.edit(el).getValue();
                        if (val && val.trim()) return val;
                    }
                } catch(e) {}
                try {
                    if (el.env && el.env.editor) {
                        const val = el.env.editor.getValue();
                        if (val && val.trim()) return val;
                    }
                } catch(e) {}
            }
            // 2. Monaco
            if (window.monaco?.editor?.getModels?.()?.length) {
                try { return window.monaco.editor.getModels()[0].getValue() || ""; } catch(e) {}
            }
            // 3. CodeMirror
            const cm = document.querySelector(".CodeMirror");
            if (cm && cm.CodeMirror) {
                try { return cm.CodeMirror.getValue() || ""; } catch(e) {}
            }
            // 4. Regular textarea (exclude Ace's 1x1 hidden input textarea)
            const tas = [...document.querySelectorAll("textarea")].filter(el => !el.classList.contains("ace_text-input"));
            for (const ta of tas) {
                const r = ta.getBoundingClientRect();
                if (r.width > 50 && r.height > 50) return ta.value || "";
            }
            return "";
        }
    """
    try:
        frames_to_check = [page]
        try:
            frames_to_check.extend(page.frames)
        except Exception:
            pass
        for frame in frames_to_check:
            try:
                txt = frame.evaluate(get_code_js)
                if txt and txt.strip():
                    return _normalize_code_indentation(txt.strip())
            except Exception:
                continue
    except Exception:
        pass
    return ""


def type_solution_code(page, code: str, language: str = ""):
    """Insert solution into Ambiance editor cleanly with zero auto-indent drift, zero line collisions, and zero duplicate braces."""
    print("[+] Typing solution code into editor...")
    try:
        code = _normalize_code_indentation(code)
        
        # AUTO-PREPEND JAVA IMPORTS
        if "java" in language.lower() and "import java.util" not in code:
            code = "import java.util.*;\nimport java.util.stream.*;\n" + code

        if not code.strip():
            print("[!] ERROR: Code is empty")
            return False

        # Always copy code to Windows clipboard first as a safety net
        copy_to_clipboard(code)

        # Collect all frames to inspect (main page + all child frames)
        all_frames = [page]
        try:
            all_frames.extend(page.frames)
        except Exception:
            pass

        # -------------------------------------------------------------
        # STEP 1: Disable Ace auto-indent, auto-pairing, & outdent rules
        # -------------------------------------------------------------
        disable_behaviors_js = """
        () => {
            let count = 0;
            const aceEls = document.querySelectorAll('.ace_editor');
            aceEls.forEach(el => {
                let ed = el.env ? el.env.editor : null;
                if (!ed && window.ace && typeof window.ace.edit === 'function') {
                    try { ed = window.ace.edit(el); } catch(e) {}
                }
                if (ed) {
                    try {
                        ed.setBehavioursEnabled(false);
                        if (ed.session && ed.session.getMode()) {
                            let m = ed.session.getMode();
                            m.getNextLineIndent = function() { return ""; };
                            m.checkOutdent = function() { return false; };
                            m.autoOutdent = function() {};
                            let proto = Object.getPrototypeOf(m);
                            if (proto) {
                                proto.getNextLineIndent = function() { return ""; };
                                proto.checkOutdent = function() { return false; };
                                proto.autoOutdent = function() {};
                            }
                        }
                        count++;
                    } catch(e) {}
                }
            });
            return count;
        }
        """
        for frame in all_frames:
            try:
                frame.evaluate(disable_behaviors_js)
            except Exception:
                pass

        # -------------------------------------------------------------
        # STEP 2: Method A - Direct Ace API Injection (Instant & exact)
        # -------------------------------------------------------------
        inject_js = """
        (newCode) => {
            let setAny = false;
            const aceEls = document.querySelectorAll('.ace_editor');
            aceEls.forEach(el => {
                let ed = el.env ? el.env.editor : null;
                if (!ed && window.ace && typeof window.ace.edit === 'function') {
                    try { ed = window.ace.edit(el); } catch(e) {}
                }
                if (ed) {
                    try {
                        ed.setValue(newCode, -1);
                        ed.clearSelection();
                        if (ed.getSession()) {
                            ed.getSession().setValue(newCode);
                            try { ed.getSession()._emit('change', {}); } catch(e) {}
                        }
                        setAny = true;
                    } catch(e) {}
                }
            });
            return setAny;
        }
        """
        read_js = """
        () => {
            const aceEls = document.querySelectorAll('.ace_editor');
            for (const el of aceEls) {
                try {
                    let ed = el.env ? el.env.editor : null;
                    if (!ed && window.ace && typeof window.ace.edit === 'function') {
                        try { ed = window.ace.edit(el); } catch(e) {}
                    }
                    if (ed) {
                        let v = ed.getValue();
                        if (v && v.trim()) return v;
                    }
                } catch(e) {}
            }
            return "";
        }
        """
        for frame in all_frames:
            try:
                frame.evaluate(inject_js, code)
            except Exception:
                pass

        # Verify whether direct API set the code
        for frame in all_frames:
            try:
                cur = frame.evaluate(read_js)
                if cur and cur.strip() == code.strip():
                    print(f"[+] Code cleanly set in Ambiance editor via Ace API ({len(code.splitlines())} lines).")
                    try:
                        page.bring_to_front()
                        inp = frame.locator(".ace_editor textarea.ace_text-input").first
                        if inp.count() > 0:
                            inp.focus()
                    except Exception:
                        pass
                    return True
            except Exception:
                continue

        # -------------------------------------------------------------
        # STEP 3: Method B - Physical Typing with Zero-Indentation Drift
        # -------------------------------------------------------------
        print(f"[*] Physically typing {len(code.splitlines())} lines into Ambiance editor...")
        page.bring_to_front()

        # Focus editor via JS first
        focus_js = """
        () => {
            const aceEls = document.querySelectorAll('.ace_editor');
            for (const el of aceEls) {
                let ed = el.env ? el.env.editor : null;
                if (!ed && window.ace && typeof window.ace.edit === 'function') {
                    try { ed = window.ace.edit(el); } catch(e) {}
                }
                if (ed) {
                    try { ed.focus(); return true; } catch(e) {}
                }
            }
            return false;
        }
        """
        for frame in all_frames:
            try:
                if frame.evaluate(focus_js):
                    break
            except Exception:
                pass

        # Also focus the Ace input element
        ace_input = page.locator('.ace_editor textarea.ace_text-input').first
        if ace_input.count() > 0:
            try:
                ace_input.focus()
                time.sleep(0.05)
            except Exception:
                pass

        # Clear existing starter code thoroughly
        page.keyboard.press("Control+A")
        time.sleep(0.05)
        page.keyboard.press("Backspace")
        time.sleep(0.05)

        # Type lines verbatim
        lines = code.splitlines()
        for i, line in enumerate(lines):
            if line:
                page.keyboard.type(line, delay=TYPE_DELAY_MS)
            if i < len(lines) - 1:
                page.keyboard.press("Enter")

        print(f"[+] Code cleanly typed into Ambiance editor ({len(lines)} lines).")
        return True

    except Exception as e:
        print(f"[!] Error in typing workflow: {e}")
        return False


def click_run_button(page):
    """Click Run button with robustness."""
    print("[+] Clicking Run button...")
    try:
        time.sleep(2)  # Ensure UI is responsive
        page.bring_to_front()

        # Primary path: visible Run button near Ambiance editor.
        run_btn = page.locator("button:visible:has-text('Run')").first
        if run_btn.count() > 0:
            run_btn.scroll_into_view_if_needed()
            run_btn.click()
            print("[+] Run button clicked")
            time.sleep(2)
            return True

        try:
            button = page.query_selector("button:has-text('Run'), button:has-text('Submit')")
            if button:
                button.click()
                print("[+] Run button clicked")
                time.sleep(2)
                return True
        except Exception:
            pass

        try:
            result = page.evaluate(
                """
                () => {
                    const buttons = document.querySelectorAll('button');
                    for (const btn of buttons) {
                        const text = (btn.textContent || '').toLowerCase();
                        if ((text.includes('run') || text.includes('submit') || text.includes('execute'))
                            && btn.offsetParent !== null) {
                            btn.click();
                            return 'clicked';
                        }
                    }
                    return 'not_found';
                }
                """
            )
            if result == "clicked":
                print("[+] Run button clicked via JavaScript")
                time.sleep(2)
                return True
        except Exception:
            pass

        try:
            submit = page.query_selector("input[type='submit']")
            if submit:
                submit.click()
                print("[+] Submit button clicked")
                time.sleep(2)
                return True
        except Exception:
            pass

        print("[!] Could not find or click Run button")
        return False
    except Exception as e:
        print(f"[!] Error clicking Run button: {e}")
        return False


def solve_captcha(page):
    """Detect and solve CAPTCHA if it appears using Pytesseract."""
    if not pytesseract:
        return False
        
    try:
        # Common SkillRack captcha selectors
        captcha_selectors = [
            "img[src*='captcha']",
            "#captchaImg",
            ".captcha-image",
            "img[alt*='captcha']"
        ]
        
        captcha_img = None
        for sel in captcha_selectors:
            el = page.locator(sel).first
            if el.is_visible(timeout=200):
                captcha_img = el
                break
        
        if captcha_img:
            print("[+] CAPTCHA detected. Solving...")
            screenshot_path = "captcha_temp.png"
            captcha_img.screenshot(path=screenshot_path)
            
            # Use Pytesseract to read the text
            captcha_text = pytesseract.image_to_string(Image.open(screenshot_path)).strip().replace(" ", "")
            print(f"[+] CAPTCHA solved: {captcha_text}")
            
            # Find input field
            input_selectors = [
                "input[name*='captcha']",
                "#captchaInput",
                "input[placeholder*='captcha']",
                "input.captcha-input"
            ]
            
            for i_sel in input_selectors:
                i_el = page.locator(i_sel).first
                if i_el.is_visible(timeout=500):
                    i_el.fill(captcha_text)
                    time.sleep(0.5)
                    return True
    except Exception as e:
        print(f"[!] CAPTCHA solving error: {e}")
    return False


def click_next_button(page):
    """Click Next button to move to the next problem."""
    print("[+] Looking for Next button...")
    try:
        # Common selectors for Next/Proceed buttons on SkillRack
        next_selectors = [
            "button:has-text('Next')",
            "a:has-text('Next')",
            "button:has-text('Proceed')",
            "a:has-text('Proceed')",
            "button:has-text('Submit & Next')",
            ".next-problem-btn"
        ]
        
        for sel in next_selectors:
            btn = page.locator(sel).first
            if btn.is_visible(timeout=3000):
                print(f"[+] Clicking Next button ({sel})")
                btn.click()
                time.sleep(3) # Wait for page load
                return True
                
        # Fallback: look for generic 'Next' text in buttons/links
        result = page.evaluate("""
            () => {
                const elements = [...document.querySelectorAll('button, a, span')];
                const next = elements.find(el => 
                    /next|proceed|forward/i.test(el.textContent) && 
                    el.offsetParent !== null
                );
                if (next) {
                    next.click();
                    return true;
                }
                return false;
            }
        """)
        if result:
            print("[+] Clicked Next via JavaScript search")
            time.sleep(3)
            return True
            
        print("[!] Next button not found.")
        return False
    except Exception as e:
        print(f"[!] Error clicking Next: {e}")
        return False


def copy_to_clipboard(text: str) -> bool:
    try:
        subprocess.run("clip", input=text, text=True, check=True, shell=True)
        return True
    except Exception:
        return False


def get_error_line_from_result(page):
    """
    Returns traceback line number if Skillrack shows a Python traceback, else None.
    """
    try:
        body = page.inner_text("body")
    except Exception:
        return None
    if not body:
        return None

    if "code did not pass the execution" not in body.lower():
        return None

    m = re.search(r"line\s+(\d+)", body, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def copy_failed_line_if_any(page, code: str):
    # Give the result panel a moment to render traceback text.
    time.sleep(2)
    line_no = get_error_line_from_result(page)
    if not line_no:
        return

    lines = code.replace("\r", "").split("\n")
    if line_no < 1 or line_no > len(lines):
        print(f"[!] Error at line {line_no}, but line is outside generated code range")
        return

    failed_line = lines[line_no - 1]
    if copy_to_clipboard(failed_line):
        print(f"[+] Copied failed line {line_no} to clipboard: {failed_line}")
    else:
        print(f"[!] Could not copy failed line {line_no} to clipboard")


# ================= LLM =================
def extract_code_from_response(response: str, language: str = ""):
    """Extract code from LLM response with robust fallback strategy."""
    if not response or len(response.strip()) < 10:
        return ""
        
    lang = (language or "").lower()
    is_java = "java" in lang
    
    # 1. Look for markdown code blocks
    code_block_pattern = re.compile(r"```[a-zA-Z]*\n?([\s\S]*?)```", re.MULTILINE)
    blocks = code_block_pattern.findall(response)
    if blocks:
        # Take largest block
        largest = max(blocks, key=len)
        if len(largest.strip()) > 10:
            return largest.strip()
            
    # 2. Heuristic extraction for Java (look for imports/class)
    is_c = bool(re.search(r"\bC\b|^\s*c\s*\(", lang, re.IGNORECASE)) and "c++" not in lang and "cpp" not in lang
    if is_java:
        match = re.search(r"(import\s+java[\s\S]*?public\s+class\s+Hello[\s\S]*\}\s*$)", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"(public\s+class\s+Hello[\s\S]*\}\s*$)", response, re.DOTALL)
        if match:
            return match.group(1).strip()
    elif is_c:
        match = re.search(r"(#include\s*<[\s\S]*?int\s+main\s*\([^)]*\)[\s\S]*\}\s*$)", response, re.DOTALL)
        if match:
            return match.group(1).strip()

    # 3. Last resort: strip any non-code explanation text from top
    lines = response.strip().splitlines()
    code_lines = []
    in_code = False
    for line in lines:
        l = line.strip()
        if not in_code:
            # Detection of code start
            if is_java:
                if l.startswith("import ") or l.startswith("public class ") or l.startswith("class "):
                    in_code = True
            elif is_c:
                if l.startswith("#include") or "int main" in l or "void main" in l or l.startswith("char ") or l.startswith("int ") or l.startswith("//"):
                    in_code = True
            else: # Python
                if l.startswith("import ") or l.startswith("from ") or l.startswith("def ") or l.startswith("n =") or l.startswith("x =") or l.startswith("#"):
                    in_code = True
        
        if in_code:
            code_lines.append(line)
            
    if code_lines:
        extracted = "\n".join(code_lines).strip()
        if is_c and "main(" not in extracted:
            extracted = "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <math.h>\n\nint main() {\n" + extracted + "\n    return 0;\n}"
        return extracted
        
    return response.strip()


def solve_problem(problem_text: str, language: str = "", starter_code: str = "", screenshot_path: str | None = None, smart_priority: bool = False):
    """Generate solution code. The llm_adapter already handles extraction and wrapping."""
    code = generate_solution(
        problem_text,
        screenshot_path=screenshot_path,
        language=language,
        starter_code=starter_code,
        smart_priority=smart_priority,
    )
    
    if not code or len(code.strip()) < 10:
        print("[!] Failed to generate valid code")
        return ""

    if "java" in (language or "").lower():
        # Guardrail: if model drifts to Python in Java mode, force one repair pass.
        if "input(" in code or "print(" in code:
            print("[!] Detected Python-like output in Java mode. Requesting Java-only repair...")
            fix_prompt = (
                "Convert this into valid Java code only.\n"
                "Return the FULL class Hello from scratch.\n"
                f"Problem:\n{problem_text}\n\n"
                f"Bad Code:\n{code}"
            )
            code = generate_solution(fix_prompt, language=language, smart_priority=True)

    if "python" not in (language or "").lower():
        return code

    try:
        ast.parse(code)
        return code
    except SyntaxError as e:
        print(f"[!] Syntax issue detected before typing ({e.msg}). Requesting syntax-only fix...")
        fix_prompt = (
            "Fix only syntax/indentation for this Python code. Keep same intent.\n"
            "Return only one ```python``` code block.\n\n"
            f"Problem:\n{problem_text}\n\n"
            f"Code:\n```python\n{code}\n```"
        )
        repaired = extract_code_from_response(
            generate_solution(
                fix_prompt,
                screenshot_path=screenshot_path,
                language=language,
                starter_code=starter_code,
            ),
            language=language,
        )
        if not repaired.strip():
            return code
        try:
            ast.parse(repaired)
            return repaired
        except SyntaxError:
            return code


def _extract_sample_io_pairs(problem_text: str, limit: int = 2):
    text = (problem_text or "").replace("\r", "")
    if not text.strip():
        return []

    def _is_ambiguous(value: str) -> bool:
        v = value.strip().lower()
        bad_tokens = (
            "input format",
            "output format",
            "sample input",
            "sample output",
            "explanation",
            "constraints",
            "problem statement",
        )
        return (not v) or any(tok in v for tok in bad_tokens)

    pairs = []
    # Matches "Input: ... Output: ..." blocks commonly used in SkillRack.
    # Uses negative lookahead to strictly ignore "Input Format:" and "Output Format:" headers!
    pattern = re.compile(
        r'(?:Example\s*(?:Input/Output)?\s*\d*[:\s]*)?'
        r'(?<![A-Za-z])Input(?!\\s*format)\s*:\s*\n?'
        r'(.*?)\n?'
        r'(?<![A-Za-z])Output(?!\\s*format)\s*:\s*\n?'
        r'(.*?)(?=(?:Explanation\s*:|Example\s*(?:Input/Output)?\s*\d*[:\s]|(?<![A-Za-z])Input(?!\\s*format)\s*:|Max Execution|$))',
        flags=re.IGNORECASE | re.DOTALL
    )
    for m in pattern.finditer(text):
        raw_inp = m.group(1).strip()
        raw_out = m.group(2).strip()
        if _is_ambiguous(raw_inp) or _is_ambiguous(raw_out):
            continue
        if len(raw_inp) > 1000 or len(raw_out) > 1000:
            continue
        pairs.append((raw_inp, raw_out))
        if len(pairs) >= limit:
            break
    return pairs


def _code_passes_sample_checks(code: str, language: str, problem_text: str) -> bool:
    if "python" not in (language or "").lower():
        return True
    if not ENABLE_SAMPLE_CHECK:
        return True

    pairs = _extract_sample_io_pairs(problem_text, limit=2)
    if not pairs:
        print("[*] Sample check skipped (no reliable sample pairs found)")
        return True

    for idx, (inp, expected) in enumerate(pairs, start=1):
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                input=inp,
                text=True,
                capture_output=True,
                timeout=SAMPLE_CHECK_TIMEOUT_SEC,
            )
        except Exception as e:
            print(f"[!] Sample check #{idx} failed to execute: {e}")
            return False

        if result.returncode != 0:
            err_snippet = (result.stderr or "").strip().splitlines()
            last_err = err_snippet[-1] if err_snippet else "Unknown error"
            print(f"[!] Sample check #{idx} runtime error ({last_err}); retrying with repaired code.")
            return False

        actual_lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        want_lines = [l.strip() for l in expected.strip().splitlines() if l.strip()]
        if actual_lines != want_lines:
            print(f"[!] Sample check #{idx} mismatch: got '{' '.join(actual_lines)}' expected '{' '.join(want_lines)}'")
            return False

    print("[+] Sample I/O checks passed")
    return True


def solve_problem_with_retry(problem_text: str, language: str = "", starter_code: str = "", screenshot_path: str | None = None, max_attempts: int = 3, smart_priority: bool = False):
    """Generate solution with validation retries until valid code is obtained."""
    last_code = ""
    fallback_code = ""

    for attempt in range(max_attempts):
        print(f"[*] LLM attempt {attempt + 1}/{max_attempts}")

        try:
            code = solve_problem(problem_text, language, starter_code, screenshot_path=screenshot_path, smart_priority=smart_priority)
            if not code or len(code) < 10:
                continue
            # Keep a non-empty candidate so we can still type something if strict checks fail.
            if not fallback_code:
                fallback_code = code
        except Exception as e:
            print(f"[!] LLM call failed on attempt {attempt + 1}: {e}")
            continue

        # Step 1: Syntax check for Python
        if "python" in language.lower():
            try:
                ast.parse(code)
                print("[+] Syntax check passed")
                fallback_code = code
            except SyntaxError as e:
                print(f"[!] Syntax failed ({e.msg}), retrying...")
                continue

        # Step 2: Basic sanity checks
        if len(code) < 20:
            print(f"[!] Code too small ({len(code)} chars), retrying...")
            continue

        # Step 3: Avoid obvious junk/incomplete code
        if "TODO" in code or ("pass" in code and len(code.split("\n")) < 5):
            print("[!] Incomplete code detected, retrying...")
            continue

        # Step 4: Validate against sample Input/Output when available.
        if not _code_passes_sample_checks(code, language, problem_text):
            print("[!] Sample validation failed. Trying one auto-repair pass...")
            try:
                repaired_response = generate_repair(
                    problem_text,
                    code,
                    "Sample Input/Output validation failed. Fix logic and output formatting exactly.",
                )
                repaired_code = extract_code_from_response(repaired_response, language=language)
                if repaired_code and len(repaired_code) >= 20:
                    if "python" in language.lower():
                        ast.parse(repaired_code)
                    if _code_passes_sample_checks(repaired_code, language, problem_text):
                        last_code = repaired_code
                        print(f"[+] Auto-repair fixed solution on attempt {attempt + 1}")
                        break
            except Exception:
                pass
            print("[!] Sample validation failed, retrying...")
            continue

        last_code = code
        print(f"[+] Valid code generated on attempt {attempt + 1}")
        break

    if last_code:
        return last_code

    # Fail-open fallback: return best syntactically valid candidate so typing continues.
    if fallback_code:
        print(f"[!] All {max_attempts} strict checks failed. Using fallback candidate.")
        return fallback_code

    print(f"[!] All {max_attempts} attempts failed.")
    return last_code


# ================= MAIN AGENT =================
def run():
    def navigate_with_retries(page, url, attempts=3):
        last_err = None
        for i in range(1, attempts + 1):
            try:
                print(f"[*] Navigating to {url} (attempt {i}/{attempts})...")
                page.goto(url, timeout=120000, wait_until="domcontentloaded")
                page.wait_for_load_state("networkidle", timeout=60000)
                return True
            except TimeoutError as te:
                print(f"[!] Navigation attempt {i} timed out: {te}")
                last_err = te
                time.sleep(1 + i)
            except Exception as e:
                print(f"[!] Navigation attempt {i} failed: {e}")
                last_err = e
                time.sleep(1 + i)

        print("[!] All navigation attempts failed")
        if last_err:
            print(f"    Last error: {last_err}")
        return False

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS, args=["--start-maximized"])
        ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        context = browser.new_context(
            user_agent=ua,
            ignore_https_errors=True,
            no_viewport=True,
        )
        # Neutralize SkillRack's anti-paste blockers while allowing Ace editor to receive paste
        context.add_init_script("""
            document.onpaste = null;
            window.onpaste = null;
            ['copy', 'cut', 'paste'].forEach(evt => {
                window.addEventListener(evt, (e) => {
                    if (e.target && (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT')) {
                        return;
                    }
                }, false);
            });
        """)
        page = context.new_page()

        print("\n===== SKILLRACK AUTOMATION =====")
        
        print("[+] Opening Skillrack in browser window...")
        # Navigate early so user can see the page and login
        ok = navigate_with_retries(page, "https://skillrack.com/faces/candidate/codeprogram.xhtml", attempts=2)
        
        # Check for LLM service (respect configured provider)
        ollama_ok = False
        if LLM_PROVIDER == "ollama":
            print("[+] LLM provider: Ollama (local).")
        elif LLM_PROVIDER == "groq":
            print("[+] LLM provider: Groq (cloud).")
        elif GROQ_API_KEY:
            print("[+] LLM provider: auto -> Groq available (cloud first).")
        else:
            print("[+] LLM provider: auto -> Ollama fallback.")

        if LLM_PROVIDER == "ollama" or (LLM_PROVIDER == "auto" and not GROQ_API_KEY):
            try:
                import requests
                requests.get(f"{os.getenv('OLLAMA_URL', 'http://localhost:11434')}/api/tags", timeout=2)
                print("[+] Ollama service detected.")
                ollama_ok = True
            except Exception:
                print("\n[!] WARNING: Ollama service is NOT running.")
                print("    Please start Ollama (run 'ollama serve' or open the Ollama app).")
                if LLM_PROVIDER == "ollama":
                    print("    You selected local mode; start Ollama and retry.\n")
                else:
                    print("    Alternatively, set GROQ_API_KEY for cloud mode.\n")
        
        # Set target solved problems (default: 200, configurable via TARGET_SOLVED env)
        target_solved = int(os.getenv("TARGET_SOLVED", "200"))
            
        print(f"[*] Target: {'Infinite' if target_solved == 0 else target_solved} problems")
        
        if not ok:
            print("[!] Could not reach SkillRack automatically. Please check your connection in the browser.")

        print("\n[!] MANUAL STEP REQUIRED:")
        print("    1. Login to SkillRack manually in the browser window.")
        print("    2. Navigate to the problem page or captcha screen.")
        print("\nPress Enter in terminal when you are ready to start solving...\n")
        input(">>> ")

        # Final check for LLM before entering the solving loop
        if (LLM_PROVIDER == "ollama" or (LLM_PROVIDER == "auto" and not GROQ_API_KEY)) and not ollama_ok:
            print("[*] Waiting for Ollama service to be ready...")
            while not ollama_ok:
                try:
                    import requests
                    requests.get(f"{os.getenv('OLLAMA_URL', 'http://localhost:11434')}/api/tags", timeout=2)
                    print("[+] Ollama service detected. Starting solving loop...")
                    ollama_ok = True
                except Exception:
                    time.sleep(2)

        problem_count = 0
        last_problem_id = None
        stuck_count = 0

        while True:
            # 0. Detect if we are stuck on the same problem
            try:
                current_id = page.evaluate("document.querySelector('.programid-label, #programID')?.innerText || ''")
                if current_id and current_id == last_problem_id:
                    stuck_count += 1
                    if stuck_count > 1: # Already tried once and failed to move on
                        print(f"[!] STUCK on {current_id}. Forcing NEXT...")
                        click_next_button(page)
                        stuck_count = 0
                        time.sleep(3)
                        continue
                else:
                    last_problem_id = current_id
                    stuck_count = 0
            except Exception:
                pass
            if target_solved > 0 and problem_count >= target_solved:
                print(f"\n[+] Successfully solved target of {target_solved} problems. Stopping.")
                break
                
            problem_count += 1
            print(f"\n{'=' * 50}")
            print(f"[*] Problem #{problem_count}")
            print(f"{'=' * 50}")

            # Check for Captcha first
            solve_captcha(page)

            print("[*] Waiting for problem page to load...")
            page.wait_for_load_state("domcontentloaded")
            time.sleep(2) 

            print("[+] Reading problem text...")
            try:
                language = get_selected_language(page)
                starter_code = get_editor_starter_code(page)
                if language:
                    print(f"[*] Detected language: {language}")
                else:
                    print(f"[!] WARNING: Could not detect language, assuming Python3")
                    language = "Python3"
                if starter_code:
                    print(f"[*] Captured starter code ({len(starter_code)} chars) from Ambiance editor")
                else:
                    print(f"[*] No starter code found (fresh problem)")

                # --- HYBRID TEXT EXTRACTION ---
                print("[*] Gathering problem text from multiple sources...")
                ocr_text = extract_text_via_ocr(page)
                raw_web_text = extract_problem_text(page)
                web_text = compact_problem_text(raw_web_text)
                
                # Merge sources: Web text provides the skeleton, OCR provides the visual proof
                text = f"WEB TEXT:\n{web_text}\n\nOCR TEXT (MAY HAVE TYPOS):\n{ocr_text}"
                
                print(f"[*] Combined problem text size: {len(text)} chars")
                
                # ... (rest of the solving logic)
                
                print("\n[+] Sending to model for solution...")
                
                # SOLVE WITH RETRY (Max 3 attempts)
                solution = solve_problem_with_retry(text, language=language, starter_code=starter_code, max_attempts=3)
                
                print("\n===== AI SOLUTION =====\n")
                print(solution) # Show full code for review
                print(f"[*] Solution size: {len(solution)} chars")

                if not solution:
                    print("[!] No solution generated. Skipping.")
                    continue
                
                # Type the generated solution into the editor
                type_ok = type_solution_code(page, solution, language=language)
                if not type_ok:
                    print("[!] CRITICAL: Failed to insert code. Please paste manually.")

                while True:
                    print("\n[!] WORKFLOW PAUSED [!]")
                    print(">>> [ENTER]   - Move to NEXT problem")
                    print(">>> [r]       - RETRY (AI will try a different approach)")
                    print(">>> [f]       - FIX (Paste an error message for AI to repair)")
                    print(">>> [s]       - SKIP this problem")
                    
                    choice = (input(">>> ") or "").strip().lower()
                    
                    if choice == "r":
                        print("[*] Retrying with different logic (using SMART model)...")
                        solution = solve_problem_with_retry(text, language=language, starter_code=starter_code, max_attempts=1, smart_priority=True)
                        if solution:
                            print("\n===== NEW AI SOLUTION =====\n")
                            print(solution)
                            type_solution_code(page, solution, language=language)
                        continue # Stay in the pause menu
                    
                    if choice == "f":
                        print("[*] Taking screenshot of browser to capture error...")
                        try:
                            ss_path = "error_screenshot.png"
                            page.screenshot(path=ss_path)
                            print("[*] Running OCR on error screenshot...")
                            error_msg = ""
                            if pytesseract:
                                from PIL import Image as PILImage
                                error_msg = pytesseract.image_to_string(PILImage.open(ss_path)).strip()
                                # Clean up OCR noise
                                error_msg = re.sub(r'\n{3,}', '\n\n', error_msg)
                            if error_msg and len(error_msg) > 20:
                                print(f"[+] OCR captured error ({len(error_msg)} chars)")
                                print(f"--- ERROR TEXT ---\n{error_msg[:500]}\n------------------")
                            else:
                                print("[!] OCR couldn't read error. Enter it manually:")
                                error_msg = input("Error: ")
                        except Exception as ocr_err:
                            print(f"[!] Screenshot/OCR failed: {ocr_err}. Enter error manually:")
                            error_msg = input("Error: ")
                        
                        print("[*] Repairing code...")
                        repair_response = generate_repair(text, solution, error_msg, language=language)
                        if repair_response:
                            repaired_code = extract_code_from_response(repair_response, language=language)
                            if repaired_code and len(repaired_code) > 20:
                                solution = repaired_code
                            else:
                                solution = repair_response
                            print("\n===== REPAIRED SOLUTION =====\n")
                            print(solution)
                            type_solution_code(page, solution, language=language)
                        continue # Stay in the pause menu

                    if choice == "s":
                        print("[*] Skipping...")
                        break
                    
                    # Default: Enter to proceed
                    print("[*] Starting 60s safety timer...")
                    for i in range(60, -1, -1):
                        mins, secs = divmod(i, 60)
                        sys.stdout.write(f"\r[*] Next problem in: {mins:02d}:{secs:02d}  ")
                        sys.stdout.flush()
                        time.sleep(1)
                    print("\n[+] Proceeding to next problem...")
                    break
                
                # Logic to proceed to next problem
                next_ok = click_next_button(page)
                if not next_ok:
                    print("[!] Could not auto-navigate. Please manually move to next problem or solve CAPTCHA.")
                    time.sleep(2)
                    
                    # Solution logging disabled as per user request
                    # with open(f"solution_{problem_count}.txt", "w", encoding="utf-8") as f:
                    #     f.write(solution)
                    # print(f"[+] Solution saved to solution_{problem_count}.txt")
                    print("\n[*] Waiting for next problem...")
                    time.sleep(3)
            except Exception as e:
                import traceback
                print(f"[!] Error processing problem: {e}")
                traceback.print_exc()
                print("\n[*] Continuing to next problem...")
                time.sleep(2)

        browser.close()


if __name__ == "__main__":
    run()
