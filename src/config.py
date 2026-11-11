"""
Configuration and Project Metadata for Smart Campus Assistant (RAG + Ollama).
Course: CSC-320(L) – Artificial Intelligence | Section B | Group 1
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "documents"
KB_JSON_PATH = DATA_DIR / "campus_knowledge_base.json"
KB_CSV_PATH = DATA_DIR / "campus_knowledge_base.csv"
EMBEDDINGS_CACHE_PATH = DATA_DIR / "embeddings_cache.json"
TEST_QUESTIONS_PATH = DATA_DIR / "test_questions.json"
EVAL_RESULTS_JSON_PATH = DATA_DIR / "evaluation_results.json"
EVAL_RESULTS_CSV_PATH = DATA_DIR / "evaluation_results.csv"

# Ollama Local API Configuration
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Preferred models (auto-detected at runtime based on `ollama list`, with fallback order)
PREFERRED_EMBED_MODELS = [
    "all-minilm:latest",
    "nomic-embed-text:latest",
    "nomic-embed-text",
    "all-minilm",
    "mxbai-embed-large:latest",
]

PREFERRED_LLM_MODELS = [
    "qwen2.5:0.5b",
    "qwen3:latest",
    "qwen3:0.6b",
    "qwen3:1.7b",
    "qwen2.5:1.5b",
    "qwen2.5:3b",
    "llama3.2:1b",
]

DEFAULT_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "all-minilm:latest")
DEFAULT_LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:0.5b")

# RAG Retrieval Parameters
DEFAULT_TOP_K = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.36  # Minimum score to consider passage relevant (abstains below this)
SEMANTIC_WEIGHT = 0.80               # Cosine similarity weight from dense embeddings
KEYWORD_OVERLAP_WEIGHT = 0.20        # Lexical/keyword overlap weight for exact campus terms

# Project & Group 1 Metadata (as specified in the Semester Project Proposal)
PROJECT_METADATA = {
    "course": "CSC-320(L) – Artificial Intelligence",
    "section": "Section B",
    "group": "Group 1",
    "title": "Smart Campus Assistant using RAG + LLM",
    "members": [
        {
            "sno": 1,
            "reg_no": "18352",
            "name": "Abdullah",
            "role": "Group coordination; requirements, Flask web application & full system integration",
            "module": "app.py, templates/, static/",
        },
        {
            "sno": 2,
            "reg_no": "18365",
            "name": "Talha",
            "role": "Collecting, verifying and organizing official campus information & FAQ knowledge base",
            "module": "data/campus_knowledge_base.json, data/campus_knowledge_base.csv, src/knowledge_base.py",
        },
        {
            "sno": 3,
            "reg_no": "18333",
            "name": "Aftab",
            "role": "Python programming; NLP question preparation, document cleaning & semantic retrieval logic",
            "module": "src/nlp_processor.py, src/retriever.py",
        },
        {
            "sno": 4,
            "reg_no": "18292",
            "name": "Shahzad",
            "role": "Ollama local model setup, embedding & LLM API integration, grounded answer generation",
            "module": "src/llm_generator.py",
        },
        {
            "sno": 5,
            "reg_no": "18165",
            "name": "Hafeez Khan",
            "role": "Testing, 25-question benchmark evaluation, documentation & presentation",
            "module": "src/evaluator.py, evaluate.py, data/test_questions.json",
        },
    ],
}
