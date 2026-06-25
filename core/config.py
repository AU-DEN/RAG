# core/config.py
import os

API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
PDF_SOURCE = "source.pdf"  # This must match the name of the file in your folder
MEMORY_FILE = "rag_memory.json"
MODEL_NAME = "gemini-2.5-flash-lite"