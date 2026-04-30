try:
    import easyocr
    print("easyocr: OK")
except ImportError:
    print("easyocr: MISSING")

try:
    import cv2
    print("cv2: OK")
except ImportError:
    print("cv2: MISSING")

try:
    import numpy
    print("numpy: OK")
except ImportError:
    print("numpy: MISSING")

try:
    import dotenv
    print("dotenv: OK")
except ImportError:
    print("dotenv: MISSING")

try:
    import playwright
    print("playwright: OK")
except ImportError:
    print("playwright: MISSING")

try:
    import requests
    print("requests: OK")
except ImportError:
    print("requests: MISSING")

try:
    import openai
    print("openai: OK")
except ImportError:
    print("openai: MISSING")
