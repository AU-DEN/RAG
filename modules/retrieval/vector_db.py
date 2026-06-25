# modules/retrieval/vector_db.py
import json
import os
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from core.config import MEMORY_FILE
from modules.ingestion.parser import load_source_documents, create_text_segments

print("Loading local embedding model...")
local_embedder = SentenceTransformer("all-MiniLM-L6-v2")

def initialize_vector_db() -> Tuple[List[str], np.ndarray]:
    """Loads memory from JSON or builds a new index."""
    # ... (Paste the initialization logic from your main() function here) ...
    pass

def generate_embeddings_batch(segments: List[str]) -> List[List[float]]:
    # ... (Paste your existing function here) ...
    pass

def retrieve_context(query: str, segments: List[str], embeddings: np.ndarray, top_k: int = 3) -> str:
    """Finds top chunks. The 'top_k' parameter will be altered by the loop."""
    query_vector = local_embedder.encode(query)
    scores = np.dot(embeddings, query_vector)
    best_indices = np.argsort(scores)[-top_k:][::-1]
    
    combined_context = "\n\n".join([segments[i] for i in best_indices])
    return combined_context