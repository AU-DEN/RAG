# Closed-Loop Adaptive RAG

A Retrieval-Augmented Generation (RAG) system that improves response reliability using a semantic faithfulness evaluation loop.

Unlike conventional RAG systems that generate responses after a single retrieval step, this implementation continuously evaluates the generated answer against the retrieved context. If the answer does not satisfy a semantic similarity threshold, the system automatically expands the retrieval context and regenerates the response.

This reduces hallucinations and increases factual consistency.

---

## Features

- Retrieval-Augmented Generation (RAG)
- Semantic Vector Search
- Automatic Faithfulness Evaluation
- Closed Feedback Loop
- Dynamic Context Expansion
- Gemini-based Answer Generation
- Retry Mechanism for Low-Confidence Responses

---

## Workflow

User Query
↓
Retrieve Top-K Chunks
↓
Generate Draft Answer
↓
Semantic Faithfulness Evaluation
↓
Faithfulness ≥ Threshold?
├── Yes → Return Answer
└── No
↓
Increase Retrieval Size
↓
Retrieve Again
↓
Generate Again

---

## Project Architecture

```
User Query
        │
        ▼
Vector Database
        │
        ▼
Relevant Context
        │
        ▼
LLM (Gemini)
        │
        ▼
Draft Answer
        │
        ▼
Semantic Similarity Checker
        │
 ┌──────┴────────┐
 │               │
Pass          Fail
 │               │
 ▼               ▼
Answer     Expand Retrieval
                 │
                 ▼
            Generate Again
```

---

## Repository Structure

```
modules/
    retrieval/
        vector_db.py
    generation/
        llm_client.py
    evaluation/
        faithfulness.py

main.py
```

---

## Technologies Used

- Python
- Sentence Transformers
- NumPy
- Google Gemini API
- FAISS / Vector Embeddings
- PDF Processing
- Cosine Similarity

---

## Installation

Clone the repository

```bash
git clone https://github.com/username/closed-loop-adaptive-rag.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python main.py
```

---

## Novelty

This project introduces a Closed-Loop Adaptive Retrieval mechanism.

Traditional RAG

```
Retrieve
↓

Generate
↓

Answer
```

Adaptive RAG

```
Retrieve
↓

Generate
↓

Evaluate Faithfulness
↓

Low Score?

↓

Expand Retrieval

↓

Generate Again
```

The iterative retrieval strategy allows the system to dynamically increase contextual information whenever hallucination risk is detected.

---

## Future Work

- Adaptive chunk sizing
- Multi-document retrieval
- Cross-encoder reranking
- Confidence estimation
- Knowledge graph integration

---

## License

MIT License
