"""
Local Ollama LLM Integration and Grounded Answer Generation Module
Author / Role: Shahzad (Reg No. 18292) – LLM/API integration and response testing.

Implements:
1. Local Ollama service health check & available model discovery (/api/tags)
2. Grounded RAG prompt construction with explicit source attribution
3. Answer generation via Ollama's /api/chat (with /api/generate fallback)
4. Strict abstention enforcement when retrieved passages do not support an answer
"""

import re
import time
from typing import Dict, List, Any, Optional

import requests

from src.config import (
    OLLAMA_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBED_MODEL,
    PREFERRED_EMBED_MODELS,
    PREFERRED_LLM_MODELS,
)

ABSTENTION_MESSAGE = (
    "I do not have enough verified information in the Campus Knowledge Base to answer this question. "
    "The Smart Campus Assistant only answers questions supported by official university notices and documents. "
    "For further assistance, please visit the Student One-Window Facilitation Desk (Block A, Room G-01) "
    "or call (051) 920-1101."
)

SYSTEM_PROMPT = """You are the official Smart Campus Assistant for university students (CSC-320 Artificial Intelligence Project).
Your strict rules are:
1. Answer the student's question ONLY using the facts provided in the "RETRIEVED CAMPUS PASSAGES" below.
2. Include exact details from the passages such as room numbers, timings, fee amounts, percentages, deadlines, and contact numbers.
3. Do NOT invent, assume, or extrapolate any information outside the provided passages.
4. Keep your answer clear, helpful, and concise (2 to 5 sentences or short bullet points).
5. At the end of your answer, cite the passage ID(s) used, e.g., [Source: KB-001].
6. If the retrieved passages do NOT contain the answer to the student's question, reply ONLY with:
"INSUFFICIENT_CONTEXT: The verified campus knowledge base does not contain information on this topic." """


class OllamaLLMGenerator:
    """
    Manages communication with the local Ollama server for grounded RAG response generation.
    """

    def __init__(
        self,
        llm_model: str = DEFAULT_LLM_MODEL,
        ollama_base_url: str = OLLAMA_BASE_URL,
    ):
        self.llm_model = llm_model
        self.ollama_base_url = ollama_base_url.rstrip("/")

    def check_ollama_status(self) -> Dict[str, Any]:
        """
        Checks whether the local Ollama server is running and lists installed models.
        Automatically categorizes embedding models vs. language models.
        """
        try:
            resp = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models_data = resp.json().get("models", [])
            installed_names = [m.get("name", "") for m in models_data]

            embed_models = [
                name for name in installed_names
                if any(k in name.lower() for k in ["embed", "minilm", "bge", "mxbai"])
            ]
            llm_models = [
                name for name in installed_names
                if name not in embed_models
            ]

            return {
                "online": True,
                "base_url": self.ollama_base_url,
                "installed_models": installed_names,
                "embed_models": embed_models,
                "llm_models": llm_models,
                "active_llm_model": self.llm_model,
            }
        except Exception as exc:
            return {
                "online": False,
                "base_url": self.ollama_base_url,
                "installed_models": [],
                "embed_models": [],
                "llm_models": [],
                "active_llm_model": self.llm_model,
                "error": str(exc),
            }

    @classmethod
    def auto_select_models(cls, ollama_base_url: str = OLLAMA_BASE_URL) -> Dict[str, str]:
        """
        Selects the best installed embedding model and LLM model from `ollama list`.
        """
        temp = cls(ollama_base_url=ollama_base_url)
        status = temp.check_ollama_status()
        installed = status.get("installed_models", [])

        chosen_embed = DEFAULT_EMBED_MODEL
        for candidate in PREFERRED_EMBED_MODELS:
            if candidate in installed:
                chosen_embed = candidate
                break
        else:
            if status.get("embed_models"):
                chosen_embed = status["embed_models"][0]

        chosen_llm = DEFAULT_LLM_MODEL
        for candidate in PREFERRED_LLM_MODELS:
            if candidate in installed:
                chosen_llm = candidate
                break
        else:
            if status.get("llm_models"):
                chosen_llm = status["llm_models"][0]

        return {
            "embed_model": chosen_embed,
            "llm_model": chosen_llm,
        }

    @staticmethod
    def build_context_block(passages: List[Dict[str, Any]]) -> str:
        """Formats the top retrieved passages into a numbered context block for the LLM."""
        blocks = []
        for idx, p in enumerate(passages, start=1):
            blocks.append(
                f"Passage {idx} [{p['id']}] (Similarity: {p['cosine_similarity']:.3f})\n"
                f"Category: {p['category']}\n"
                f"Title: {p['title']}\n"
                f"Information: {p['content']}\n"
                f"Verified Source: {p['source']}"
            )
        return "\n\n---\n\n".join(blocks)

    def generate_answer(
        self,
        question: str,
        retrieval_result: Dict[str, Any],
        llm_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generates a grounded answer from the retrieved passages using the local Ollama LLM,
        or abstains cleanly if the question is outside the knowledge base coverage.
        """
        active_model = llm_model or self.llm_model
        top_passages = retrieval_result.get("top_passages", [])
        is_relevant = retrieval_result.get("is_relevant", False)

        # Case 1: Retrieval component determined that no passage meets the relevance criteria
        if not is_relevant or not top_passages:
            timing = dict(retrieval_result.get("timing_ms", {}))
            timing["llm_ms"] = 0.0
            timing["total_ms"] = round(sum(timing.values()), 2)
            return {
                "question": question,
                "answer": ABSTENTION_MESSAGE,
                "abstained": True,
                "abstention_reason": retrieval_result.get(
                    "abstention_reason",
                    "No relevant campus passages met the semantic similarity threshold.",
                ),
                "sources": [],
                "retrieved_passages": top_passages,
                "nlp": retrieval_result.get("nlp", {}),
                "llm_model": active_model,
                "embed_model": retrieval_result.get("embed_model", DEFAULT_EMBED_MODEL),
                "max_cosine_similarity": retrieval_result.get("max_cosine_similarity", 0.0),
                "max_combined_score": retrieval_result.get("max_combined_score", 0.0),
                "timing_ms": timing,
            }

        # Case 2: Relevant passages found -> Construct grounded prompt for local Ollama LLM
        context_block = self.build_context_block(top_passages)
        user_prompt = (
            f"RETRIEVED CAMPUS PASSAGES:\n{context_block}\n\n"
            f"STUDENT QUESTION: {question}\n\n"
            f"Provide a direct, accurate answer based ONLY on the retrieved campus passages above, "
            f"including specific numbers, timings, room numbers, or policies mentioned:"
        )

        t_llm = time.perf_counter()
        raw_reply = ""
        try:
            resp = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json={
                    "model": active_model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 180,
                        "num_ctx": 2048,
                    },
                },
                timeout=90,
            )
            if resp.status_code == 200:
                raw_reply = resp.json().get("message", {}).get("content", "").strip()
            else:
                # Fallback to /api/generate
                gen_resp = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": active_model,
                        "system": SYSTEM_PROMPT,
                        "prompt": user_prompt,
                        "stream": False,
                        "options": {"temperature": 0.1, "num_predict": 280},
                    },
                    timeout=90,
                )
                gen_resp.raise_for_status()
                raw_reply = gen_resp.json().get("response", "").strip()
        except Exception as exc:
            # Fallback to direct extractive summary if Ollama LLM call encounters a transient error
            p0 = top_passages[0]
            raw_reply = (
                f"{p0['content']} [{p0['id']} — {p0['source']}]\n"
                f"(Note: Local LLM generation fallback activated: {exc})"
            )

        llm_ms = round((time.perf_counter() - t_llm) * 1000, 2)

        # Strip optional <think>...</think> blocks from reasoning models like Qwen 3
        cleaned_reply = re.sub(r"<think>.*?</think>", "", raw_reply, flags=re.DOTALL).strip()

        # Check if the LLM explicitly abstained via INSUFFICIENT_CONTEXT
        if "INSUFFICIENT_CONTEXT" in cleaned_reply.upper():
            timing = dict(retrieval_result.get("timing_ms", {}))
            timing["llm_ms"] = llm_ms
            timing["total_ms"] = round(sum(timing.values()), 2)
            return {
                "question": question,
                "answer": ABSTENTION_MESSAGE,
                "abstained": True,
                "abstention_reason": "The local LLM determined that the retrieved passages do not answer the specific question.",
                "sources": [],
                "retrieved_passages": top_passages,
                "nlp": retrieval_result.get("nlp", {}),
                "llm_model": active_model,
                "embed_model": retrieval_result.get("embed_model", DEFAULT_EMBED_MODEL),
                "max_cosine_similarity": retrieval_result.get("max_cosine_similarity", 0.0),
                "max_combined_score": retrieval_result.get("max_combined_score", 0.0),
                "timing_ms": timing,
            }

        # Prepare cited sources from Top-K relevant passages
        sources = [
            {
                "id": p["id"],
                "title": p["title"],
                "category": p["category"],
                "source": p["source"],
                "cosine_similarity": p["cosine_similarity"],
                "combined_score": p["combined_score"],
            }
            for p in top_passages
        ]

        # Ensure source reference appears clearly if the compact LLM omitted the bracket tag
        primary_source = top_passages[0]
        if primary_source["id"] not in cleaned_reply:
            cleaned_reply = f"{cleaned_reply}\n\n[Verified Source: {primary_source['id']} – {primary_source['source']}]"

        timing = dict(retrieval_result.get("timing_ms", {}))
        timing["llm_ms"] = llm_ms
        timing["total_ms"] = round(sum(timing.values()), 2)

        return {
            "question": question,
            "answer": cleaned_reply,
            "abstained": False,
            "abstention_reason": None,
            "sources": sources,
            "retrieved_passages": top_passages,
            "nlp": retrieval_result.get("nlp", {}),
            "llm_model": active_model,
            "embed_model": retrieval_result.get("embed_model", DEFAULT_EMBED_MODEL),
            "max_cosine_similarity": retrieval_result.get("max_cosine_similarity", 0.0),
            "max_combined_score": retrieval_result.get("max_combined_score", 0.0),
            "timing_ms": timing,
            "prompt_used": user_prompt,
        }
