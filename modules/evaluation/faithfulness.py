# modules/evaluation/faithfulness.py
from sentence_transformers import util
from modules.retrieval.vector_db import local_embedder

def check_semantic_similarity(context_chunk: str, generated_answer: str) -> float:
    """Calculates cosine similarity to ensure the answer matches the context."""
    embeddings = local_embedder.encode([context_chunk, generated_answer], convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings[0], embeddings[1])
    return cosine_scores.item()