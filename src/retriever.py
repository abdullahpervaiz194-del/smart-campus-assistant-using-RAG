"""
Semantic Embedding and RAG Retrieval Module
Author / Role: Aftab (Reg No. 18333) & Shahzad (Reg No. 18292)

Implements:
1. Dense vector embedding generation using Ollama's local REST API (/api/embed)
2. Persistent local embedding cache (to avoid recomputing unchanged passages)
3. Vector Cosine Similarity computation in NumPy
4. Top-K passage retrieval with similarity threshold & out-of-domain abstention detection
"""

import hashlib
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import requests

from src.config import (
    OLLAMA_BASE_URL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    SEMANTIC_WEIGHT,
    KEYWORD_OVERLAP_WEIGHT,
    EMBEDDINGS_CACHE_PATH,
)
from src.nlp_processor import (
    preprocess_question,
    format_passage_for_embedding,
    tokenize_and_extract_keywords,
)

# Generic words that appear across almost all campus notices and should not alone
# trigger a high lexical match without a specific topic keyword.
GENERIC_CAMPUS_WORDS = {
    "university", "campus", "student", "students", "college", "school",
    "semester", "program", "programs", "course", "courses", "office",
    "department", "rule", "rules", "policy", "information", "system",
    "available", "apply", "required", "requirement", "requirements",
    "current", "today", "best", "home", "many", "much", "number", "time",
    "day", "days", "week", "weeks", "year", "new", "official", "general",
}


class SemanticRetriever:
    """
    Embeds campus knowledge base passages via local Ollama embedding models
    and retrieves the most relevant passages for a student question using Cosine Similarity.
    """

    def __init__(
        self,
        entries: List[Dict[str, Any]],
        embed_model: str = DEFAULT_EMBED_MODEL,
        ollama_base_url: str = OLLAMA_BASE_URL,
        cache_path: Path = EMBEDDINGS_CACHE_PATH,
    ):
        self.entries = entries
        self.embed_model = embed_model
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.cache_path = cache_path
        self.passage_embeddings: Optional[np.ndarray] = None
        self.passage_token_sets: List[set] = []
        self.kb_specific_vocab: set = set()
        self._prepare_token_sets()

    @staticmethod
    def _stem_simple(word: str) -> str:
        """Lightweight suffix normalization for matching singular/plural campus terms."""
        if len(word) > 5 and word.endswith("ies"):
            return word[:-3] + "y"
        if len(word) > 4 and word.endswith("es"):
            return word[:-2]
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
        return word

    def _prepare_token_sets(self) -> None:
        """Precomputes keyword/token sets for each passage and global specific vocabulary."""
        self.passage_token_sets = []
        self.kb_specific_vocab = set()
        for entry in self.entries:
            combined_text = (
                f"{entry.get('category', '')} {entry.get('title', '')} "
                f"{entry.get('content', '')} {' '.join(entry.get('keywords', []))}"
            )
            raw_tokens = set(tokenize_and_extract_keywords(combined_text))
            stemmed_tokens = {self._stem_simple(t) for t in raw_tokens}
            all_entry_tokens = raw_tokens | stemmed_tokens
            specific_tokens = all_entry_tokens - GENERIC_CAMPUS_WORDS
            self.passage_token_sets.append(specific_tokens)
            self.kb_specific_vocab.update(specific_tokens)

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _load_cache(self) -> Dict[str, Any]:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self, cache_data: Dict[str, Any]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)

    def embed_texts_ollama(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """
        Calls Ollama's local embedding endpoint (/api/embed or fallback /api/embeddings)
        to convert text strings into numerical vector embeddings.
        """
        target_model = model or self.embed_model
        try:
            resp = requests.post(
                f"{self.ollama_base_url}/api/embed",
                json={"model": target_model, "input": texts, "keep_alive": "30m"},
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                if "embeddings" in data and len(data["embeddings"]) == len(texts):
                    return data["embeddings"]
        except requests.RequestException:
            pass

        embeddings = []
        for text in texts:
            resp = requests.post(
                f"{self.ollama_base_url}/api/embeddings",
                json={"model": target_model, "prompt": text},
                timeout=30,
            )
            resp.raise_for_status()
            embeddings.append(resp.json()["embedding"])
        return embeddings

    def build_or_load_index(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Builds or loads cached vector embeddings for all campus knowledge base entries.
        """
        start_time = time.perf_counter()
        self._prepare_token_sets()
        cache = {} if force_rebuild else self._load_cache()
        model_cache = cache.get(self.embed_model, {})

        formatted_texts = [format_passage_for_embedding(e) for e in self.entries]
        hashes = [self._hash_text(t) for t in formatted_texts]

        missing_indices = []
        missing_texts = []
        for idx, (entry, h) in enumerate(zip(self.entries, hashes)):
            key = f"{entry['id']}_{h}"
            if key not in model_cache:
                missing_indices.append(idx)
                missing_texts.append(formatted_texts[idx])

        if missing_texts:
            new_vectors = self.embed_texts_ollama(missing_texts, model=self.embed_model)
            for idx, vec in zip(missing_indices, new_vectors):
                key = f"{self.entries[idx]['id']}_{hashes[idx]}"
                model_cache[key] = vec
            cache[self.embed_model] = model_cache
            self._save_cache(cache)

        ordered_vectors = [
            model_cache[f"{entry['id']}_{h}"]
            for entry, h in zip(self.entries, hashes)
        ]
        matrix = np.array(ordered_vectors, dtype=np.float32)

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.passage_embeddings = matrix / norms

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "total_passages": len(self.entries),
            "newly_embedded": len(missing_texts),
            "embedding_dim": int(self.passage_embeddings.shape[1]) if len(self.entries) > 0 else 0,
            "embed_model": self.embed_model,
            "index_time_ms": elapsed_ms,
        }

    @staticmethod
    def compute_cosine_similarities(query_vec: np.ndarray, passage_matrix: np.ndarray) -> np.ndarray:
        """
        Computes Cosine Similarity between a query embedding vector q and all passage vectors D:
            cos(theta) = (q . d_i) / (||q|| * ||d_i||)
        """
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            return np.zeros(passage_matrix.shape[0], dtype=np.float32)
        normalized_q = query_vec / q_norm
        similarities = np.dot(passage_matrix, normalized_q)
        return np.clip(similarities, -1.0, 1.0)

    def retrieve(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Preprocesses the student's question, computes its Ollama embedding, calculates
        cosine similarity against all campus passages, and returns Top-K passages.
        """
        t0 = time.perf_counter()
        nlp_info = preprocess_question(question)
        nlp_ms = round((time.perf_counter() - t0) * 1000, 2)

        if self.passage_embeddings is None or len(self.passage_embeddings) != len(self.entries):
            self.build_or_load_index()

        # Embed the expanded question
        t_embed = time.perf_counter()
        query_for_embed = nlp_info["expanded_question"] or nlp_info["cleaned_question"] or question
        query_vec_list = self.embed_texts_ollama([query_for_embed], model=self.embed_model)[0]
        query_vec = np.array(query_vec_list, dtype=np.float32)
        embed_ms = round((time.perf_counter() - t_embed) * 1000, 2)

        # Compute Cosine Similarities
        t_ret = time.perf_counter()
        cosine_scores = self.compute_cosine_similarities(query_vec, self.passage_embeddings)

        # Specific query terms (excluding generic campus stopwords)
        specific_raw_kws = {
            w for w in nlp_info["keywords"] if w not in GENERIC_CAMPUS_WORDS
        }
        specific_expanded_kws = {
            w for w in nlp_info["expanded_keywords"] if w not in GENERIC_CAMPUS_WORDS
        }
        specific_stemmed_kws = {self._stem_simple(w) for w in specific_expanded_kws}
        query_search_terms = specific_expanded_kws | specific_stemmed_kws

        # Check how many of the user's specific raw keywords exist anywhere in the KB vocabulary
        known_raw_kws = [
            w for w in specific_raw_kws
            if w in self.kb_specific_vocab or self._stem_simple(w) in self.kb_specific_vocab
        ]
        vocab_coverage_ratio = (
            len(known_raw_kws) / len(specific_raw_kws) if specific_raw_kws else 0.0
        )

        scored_passages = []
        for idx, entry in enumerate(self.entries):
            cos_sim = float(cosine_scores[idx])
            passage_tokens = self.passage_token_sets[idx]

            matched_tokens = sorted(list(query_search_terms.intersection(passage_tokens)))
            if specific_raw_kws:
                kw_ratio = len(matched_tokens) / max(1, min(len(specific_raw_kws), 4))
                kw_ratio = min(1.0, kw_ratio)
            else:
                kw_ratio = 0.0

            combined_score = (SEMANTIC_WEIGHT * cos_sim) + (KEYWORD_OVERLAP_WEIGHT * kw_ratio)

            scored_passages.append({
                "id": entry["id"],
                "category": entry["category"],
                "title": entry["title"],
                "content": entry["content"],
                "source": entry["source"],
                "keywords": entry.get("keywords", []),
                "cosine_similarity": round(cos_sim, 4),
                "keyword_overlap": round(kw_ratio, 4),
                "combined_score": round(combined_score, 4),
                "matched_keywords": matched_tokens,
            })

        scored_passages.sort(key=lambda x: (x["combined_score"], x["cosine_similarity"]), reverse=True)
        top_passages = scored_passages[:top_k]

        max_cosine = top_passages[0]["cosine_similarity"] if top_passages else 0.0
        max_combined = top_passages[0]["combined_score"] if top_passages else 0.0
        top_matched_kws = top_passages[0]["matched_keywords"] if top_passages else []

        # Relevance & Abstention Decision Logic
        is_relevant = True
        abstention_reason = None

        if max_combined < similarity_threshold and max_cosine < similarity_threshold:
            is_relevant = False
            abstention_reason = (
                f"Highest semantic similarity ({max_cosine:.3f}) is below the minimum threshold ({similarity_threshold:.2f})."
            )
        elif len(known_raw_kws) == 0 and max_cosine < 0.55:
            is_relevant = False
            abstention_reason = (
                "None of the specific topic keywords in the question exist in the verified campus knowledge base."
            )
        elif vocab_coverage_ratio < 0.25 and not nlp_info["has_campus_domain_signal"] and max_cosine < 0.52:
            is_relevant = False
            abstention_reason = (
                f"Low campus vocabulary coverage ({vocab_coverage_ratio:.0%}) and low semantic similarity ({max_cosine:.3f})."
            )

        retrieval_ms = round((time.perf_counter() - t_ret) * 1000, 2)

        return {
            "nlp": nlp_info,
            "top_passages": top_passages,
            "max_cosine_similarity": round(max_cosine, 4),
            "max_combined_score": round(max_combined, 4),
            "vocab_coverage_ratio": round(vocab_coverage_ratio, 4),
            "known_topic_keywords": known_raw_kws,
            "similarity_threshold": similarity_threshold,
            "is_relevant": is_relevant,
            "abstention_reason": abstention_reason,
            "embedding_dim": int(query_vec.shape[0]),
            "embed_model": self.embed_model,
            "timing_ms": {
                "nlp_ms": nlp_ms,
                "embedding_ms": embed_ms,
                "retrieval_ms": retrieval_ms,
            },
        }
