"""
CLI Evaluation Script for Smart Campus Assistant (CSC-320(L) Group 1)
Author / Role: Hafeez Khan (Reg No. 18165)

Usage:
    .venv\\Scripts\\python.exe evaluate.py
"""

from src.knowledge_base import CampusKnowledgeBase
from src.retriever import SemanticRetriever
from src.llm_generator import OllamaLLMGenerator
from src.evaluator import RAGEvaluator


def main() -> None:
    print("=" * 72)
    print(" SMART CAMPUS ASSISTANT (RAG + OLLAMA) – 25-QUESTION EVALUATION SUITE")
    print(" Course: CSC-320(L) Artificial Intelligence | Section B | Group 1")
    print("=" * 72)

    models = OllamaLLMGenerator.auto_select_models()
    embed_model = models["embed_model"]
    llm_model = models["llm_model"]
    print(f"[*] Detected Local Embedding Model : {embed_model}")
    print(f"[*] Detected Local LLM Model       : {llm_model}")

    kb = CampusKnowledgeBase()
    print(f"[*] Loaded Verified Knowledge Base : {len(kb.entries)} passages across {len(kb.get_categories())} categories")

    retriever = SemanticRetriever(entries=kb.entries, embed_model=embed_model)
    idx_stats = retriever.build_or_load_index()
    print(
        f"[*] Vector Index Ready             : {idx_stats['total_passages']} vectors "
        f"(dim={idx_stats['embedding_dim']}) in {idx_stats['index_time_ms']} ms"
    )

    generator = OllamaLLMGenerator(llm_model=llm_model)
    evaluator = RAGEvaluator(retriever=retriever, generator=generator)

    print("\n[*] Running 25 test questions (19 in-scope + 6 out-of-scope abstention tests)...")
    report = evaluator.run_evaluation()
    summary = report["summary"]

    print("\n" + "-" * 72)
    print(" EVALUATION SUMMARY METRICS")
    print("-" * 72)
    print(f" Total Test Questions              : {summary['total_questions']} ({summary['in_scope_questions']} In-Scope, {summary['out_of_scope_questions']} Out-of-Scope)")
    print(f" Retrieval Hit@1 Accuracy          : {summary['retrieval_hit_at_1_rate']}%")
    print(f" Retrieval Hit@3 Relevance         : {summary['retrieval_hit_at_3_rate']}%")
    print(f" In-Scope Grounded Answer Accuracy : {summary['in_scope_answer_accuracy']}%")
    print(f" Out-of-Scope Abstention Accuracy  : {summary['out_of_scope_abstention_accuracy']}%")
    print(f" Overall System Accuracy           : {summary['overall_system_accuracy']}%")
    print(f" Average Embedding Latency         : {summary['avg_embedding_time_ms']} ms")
    print(f" Average LLM Generation Latency    : {summary['avg_llm_time_ms']} ms")
    print(f" Average End-to-End Response Time  : {summary['avg_total_time_ms']} ms")
    print("-" * 72)
    print(" Saved detailed reports to:")
    print("   - data/evaluation_results.json")
    print("   - data/evaluation_results.csv")
    print("=" * 72)


if __name__ == "__main__":
    main()
