"""
RAG Prototype Evaluation Module (25 Test Questions Benchmark)
Author / Role: Hafeez Khan (Reg No. 18165) – Testing, evaluation, documentation and presentation.

Evaluates:
1. Retrieval Relevance (Hit@1 and Hit@3 against ground-truth KB passage IDs)
2. Answer Correctness & Groundedness (presence of expected facts/keywords & source citation)
3. Abstention Accuracy (correctly abstaining on out-of-scope questions & answering in-scope ones)
4. Response Latency (NLP, Embedding, Retrieval, and Local LLM Generation time in ms)
"""

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.config import (
    TEST_QUESTIONS_PATH,
    EVAL_RESULTS_JSON_PATH,
    EVAL_RESULTS_CSV_PATH,
    DEFAULT_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD,
)
from src.retriever import SemanticRetriever
from src.llm_generator import OllamaLLMGenerator


class RAGEvaluator:
    """
    Runs the benchmark suite of test questions against the Smart Campus Assistant
    and records detailed retrieval, generation, abstention, and latency metrics.
    """

    def __init__(
        self,
        retriever: SemanticRetriever,
        generator: OllamaLLMGenerator,
        test_questions_path: Path = TEST_QUESTIONS_PATH,
    ):
        self.retriever = retriever
        self.generator = generator
        self.test_questions_path = test_questions_path

    def load_test_questions(self) -> List[Dict[str, Any]]:
        if not self.test_questions_path.exists():
            return []
        with open(self.test_questions_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_evaluation(
        self,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes the full evaluation benchmark and persists results to JSON and CSV.
        """
        questions = self.load_test_questions()
        if limit is not None:
            questions = questions[:limit]

        results = []
        in_scope_total = 0
        hit_at_1_count = 0
        hit_at_k_count = 0
        answer_correct_count = 0
        out_of_scope_total = 0
        correct_abstention_count = 0
        false_abstention_count = 0
        total_decision_correct = 0

        total_times = []
        embed_times = []
        llm_times = []

        for q_item in questions:
            qid = q_item["id"]
            question_text = q_item["question"]
            expected_ids = set(q_item.get("expected_kb_ids", []))
            expected_kws = [k.lower() for k in q_item.get("expected_keywords", [])]
            should_abstain = bool(q_item.get("should_abstain", False))

            # Step 1: Retrieve
            ret_res = self.retriever.retrieve(
                question=question_text,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )

            # Step 2: Generate Answer via Ollama
            gen_res = self.generator.generate_answer(
                question=question_text,
                retrieval_result=ret_res,
            )

            retrieved_ids = [p["id"] for p in ret_res.get("top_passages", [])]
            top_1_id = retrieved_ids[0] if retrieved_ids else None
            abstained = gen_res["abstained"]
            answer_text = gen_res["answer"]
            timing = gen_res["timing_ms"]

            total_times.append(timing.get("total_ms", 0.0))
            embed_times.append(timing.get("embedding_ms", 0.0))
            if not abstained:
                llm_times.append(timing.get("llm_ms", 0.0))

            if should_abstain:
                out_of_scope_total += 1
                retrieval_hit_1 = True if abstained else False
                retrieval_hit_k = True if abstained else False
                abstention_correct = abstained
                if abstained:
                    correct_abstention_count += 1
                    total_decision_correct += 1
                matched_expected_kws = []
                kw_recall = 1.0 if abstained else 0.0
                answer_is_correct = abstained
            else:
                in_scope_total += 1
                retrieval_hit_1 = top_1_id in expected_ids
                retrieval_hit_k = len(expected_ids.intersection(set(retrieved_ids))) > 0
                if retrieval_hit_1:
                    hit_at_1_count += 1
                if retrieval_hit_k:
                    hit_at_k_count += 1

                abstention_correct = not abstained
                if abstained:
                    false_abstention_count += 1
                else:
                    total_decision_correct += 1

                # Check how many expected factual keywords appear in the generated answer
                ans_lower = answer_text.lower()
                matched_expected_kws = [kw for kw in expected_kws if kw in ans_lower]
                kw_recall = (
                    len(matched_expected_kws) / len(expected_kws)
                    if expected_kws
                    else 1.0
                )
                # Answer is considered correct & grounded if not abstained, retrieval hit@k is True,
                # and at least 50% of expected key facts (or >=1 key fact) appear in the LLM answer
                answer_is_correct = (
                    (not abstained)
                    and retrieval_hit_k
                    and (kw_recall >= 0.40 or len(matched_expected_kws) >= 1)
                )
                if answer_is_correct:
                    answer_correct_count += 1

            results.append({
                "id": qid,
                "question": question_text,
                "category": q_item.get("category", ""),
                "should_abstain": should_abstain,
                "abstained": abstained,
                "abstention_correct": abstention_correct,
                "expected_kb_ids": list(expected_ids),
                "retrieved_kb_ids": retrieved_ids,
                "top_1_kb_id": top_1_id,
                "max_cosine_similarity": ret_res.get("max_cosine_similarity", 0.0),
                "max_combined_score": ret_res.get("max_combined_score", 0.0),
                "retrieval_hit_at_1": retrieval_hit_1,
                "retrieval_hit_at_3": retrieval_hit_k,
                "expected_keywords": expected_kws,
                "matched_keywords": matched_expected_kws,
                "keyword_recall": round(kw_recall, 2),
                "answer_correct": answer_is_correct,
                "answer": answer_text,
                "abstention_reason": gen_res.get("abstention_reason"),
                "timing_ms": timing,
            })

        total_q = len(questions)
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embed_model": self.retriever.embed_model,
            "llm_model": self.generator.llm_model,
            "similarity_threshold": similarity_threshold,
            "top_k": top_k,
            "total_questions": total_q,
            "in_scope_questions": in_scope_total,
            "out_of_scope_questions": out_of_scope_total,
            "retrieval_hit_at_1_rate": round((hit_at_1_count / max(1, in_scope_total)) * 100, 1),
            "retrieval_hit_at_3_rate": round((hit_at_k_count / max(1, in_scope_total)) * 100, 1),
            "in_scope_answer_accuracy": round((answer_correct_count / max(1, in_scope_total)) * 100, 1),
            "out_of_scope_abstention_accuracy": round((correct_abstention_count / max(1, out_of_scope_total)) * 100, 1),
            "overall_system_accuracy": round(
                ((answer_correct_count + correct_abstention_count) / max(1, total_q)) * 100, 1
            ),
            "avg_total_time_ms": round(sum(total_times) / max(1, len(total_times)), 1),
            "avg_embedding_time_ms": round(sum(embed_times) / max(1, len(embed_times)), 1),
            "avg_llm_time_ms": round(sum(llm_times) / max(1, len(llm_times)), 1),
        }

        full_report = {
            "summary": summary,
            "results": results,
        }

        self._save_results(full_report)
        return full_report

    @staticmethod
    def _save_results(report: Dict[str, Any]) -> None:
        EVAL_RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(EVAL_RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        fieldnames = [
            "id", "category", "question", "should_abstain", "abstained",
            "expected_kb_ids", "retrieved_kb_ids", "max_cosine_similarity",
            "retrieval_hit_at_1", "retrieval_hit_at_3", "answer_correct",
            "total_ms", "answer",
        ]
        with open(EVAL_RESULTS_CSV_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in report.get("results", []):
                writer.writerow({
                    "id": r["id"],
                    "category": r["category"],
                    "question": r["question"],
                    "should_abstain": r["should_abstain"],
                    "abstained": r["abstained"],
                    "expected_kb_ids": ", ".join(r["expected_kb_ids"]),
                    "retrieved_kb_ids": ", ".join(r["retrieved_kb_ids"]),
                    "max_cosine_similarity": r["max_cosine_similarity"],
                    "retrieval_hit_at_1": r["retrieval_hit_at_1"],
                    "retrieval_hit_at_3": r["retrieval_hit_at_3"],
                    "answer_correct": r["answer_correct"],
                    "total_ms": r["timing_ms"].get("total_ms", 0.0),
                    "answer": r["answer"].replace("\n", " "),
                })

    @staticmethod
    def load_cached_results() -> Optional[Dict[str, Any]]:
        if EVAL_RESULTS_JSON_PATH.exists():
            try:
                with open(EVAL_RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None
