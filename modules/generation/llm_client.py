# modules/generation/llm_client.py
import google.generativeai as genai
from core.config import API_KEY, MODEL_NAME

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

def generate_draft_answer(user_query: str, context: str) -> str:
    """Sends the prompt to Gemini."""
    prompt = (
        "You are a helpful AI Tutor. Answer the user's question "
        "strictly based on the provided context. Do not use outside knowledge.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {user_query}"
    )
    response = model.generate_content(prompt)
    return response.text