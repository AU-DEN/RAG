import os
import time
import google.generativeai as genai
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Optional


API_KEY = "AIzaSyAyYtEneFYeOl5UqtKHzl9Pmbw73I5r1CM"
PDF_SOURCE = "source.pdf"
TXT_SOURCE = "source.txt"
COLLECTION_NAME = "my_knowledge_base"
MODEL_NAME = "gemini-2.5-flash-lite"
EMBEDDING_MODEL = "models/text-embedding-004"

client = QdrantClient(path="qdrant_db")

if API_KEY == "PASTE_YOUR_KEY_HERE":
    raise ValueError("  Configuration Error: API Key is missing.")

genai.configure(api_key=API_KEY)

def load_source_text() -> Optional[str]:
    """Loads text from TXT or PDF."""
    if os.path.exists(TXT_SOURCE):
        print(f" Loading '{TXT_SOURCE}'...")
        with open(TXT_SOURCE, "r", encoding="utf-8") as f:
            return f.read()

    if os.path.exists(PDF_SOURCE):
        print(f" Parsing '{PDF_SOURCE}'...")
        reader = PdfReader(PDF_SOURCE)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content: text += content + "\n"
        return text
    
    return None

def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]: #this is for adjusting chunk size and overlap
    """Splits text into chunks."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def setup_knowledge_base():
    """Reads file, embeds it, and saves to Qdrant."""
    if client.collection_exists(collection_name=COLLECTION_NAME):
        print(" Found existing Qdrant database. Loading...")
        return

    print("  No database found. Creating new one...")
    raw_text = load_source_text()
    if not raw_text:
        print(" No data found to process.")
        return

    chunks = split_text(raw_text)
    print(f"  Split into {len(chunks)} chunks.")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    print(" Created Qdrant collection.")
    print(" Vectorizing and saving to Qdrant...")
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=batch,
            task_type="retrieval_document"
        )
        embeddings = result['embedding']
        points = []
        for j, vector in enumerate(embeddings):
            points.append(PointStruct(
                id=i + j, 
                vector=vector, 
                payload={"text": batch[j]} 
            ))
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print(f"   Saved batch {i // batch_size + 1}...")
        time.sleep(1)

    print(" Knowledge Base Built!")

def search_qdrant(query: str):
    """Searches Qdrant for the best match."""
    query_vector = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="retrieval_query"
    )['embedding']
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3
    ).points
    context = ""
    for result in search_result:
        context += result.payload['text'] + "\n\n"
    
    return context

if __name__ == "__main__":
    setup_knowledge_base()
    
    model = genai.GenerativeModel(MODEL_NAME)
    print("\n Qdrant System Ready. Ask a question.")
    
    while True:
        question = input("\nYou: ")
        if question.lower() in ["q", "quit"]: break
        context = search_qdrant(question)
        prompt = f"""
        Answer based on this context:
        {context}
        
        Question: {question}
        """
        response = model.generate_content(prompt)
        print(f"AI: {response.text}")