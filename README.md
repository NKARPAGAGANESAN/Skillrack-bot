SkillRack Bot

SkillRack Bot is a high-performance, autonomous browser-based system designed to assist with solving coding challenges on SkillRack. It uses Playwright for browser automation and Gemini AI for code generation and iterative self-repair.

✨ Key Features
Autonomous Workflow: Handles navigation, code entry, execution, and submission automatically.
Dynamic API Key Rotation: Supports multiple Gemini API keys to manage rate limits and maintain uninterrupted execution.
Self-Healing Mechanism: Captures screenshots and logs on failure, then retries with improved solutions.
Multi-Language Support: Works with Python, Java (enforces public class Hello), and C, with structured input/output handling.
OCR Integration: Uses easyocr to extract problem statements when text selection is restricted.
🛠️ Installation & Setup
1. Install dependencies
pip install -r requirements.txt
2. Install Playwright browser
playwright install chromium
3. Configure API Keys

Create a .env file inside the project directory:

GOOGLE_API_KEY=your_api_key_here

For multiple keys, update the GEMINI_KEYS list in:

bot/llm_adapter.py

⚠️ Never upload .env or any real API keys to GitHub.

🚀 Running the Bot
python bot/main.py

📁 Project Guidelines
Keep sensitive data (API keys) in .env or secure storage
Ensure Java solutions use public class Hello
Use logs or screenshots for debugging failures

⚠️ Disclaimer

This project is intended for educational and experimental purposes only. Ensure compliance with SkillRack’s terms of service before using automation tools.
