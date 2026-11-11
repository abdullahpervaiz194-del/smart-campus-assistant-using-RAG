"""
Flask Web Application – Smart Campus Assistant using RAG + LLM (Ollama)
Author / Role: Abdullah (Reg No. 18352) – Group coordination, Flask application & system integration.

Course: CSC-320(L) – Artificial Intelligence | Section B | Group 1
"""

from flask import Flask, render_template, request, jsonify

from src.config import (
    PROJECT_METADATA,
    DEFAULT_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD,
)
from src.knowledge_base import CampusKnowledgeBase
from src.retriever import SemanticRetriever
from src.llm_generator import OllamaLLMGenerator
from src.evaluator import RAGEvaluator

app = Flask(__name__)

# Initialize Knowledge Base, Ollama Model Auto-Detection, Retriever, and Generator
kb = CampusKnowledgeBase()
detected_models = OllamaLLMGenerator.auto_select_models()
active_embed_model = detected_models["embed_model"]
active_llm_model = detected_models["llm_model"]

retriever = SemanticRetriever(entries=kb.entries, embed_model=active_embed_model)
generator = OllamaLLMGenerator(llm_model=active_llm_model)
evaluator = RAGEvaluator(retriever=retriever, generator=generator)

# Build or load cached embeddings on startup
try:
    index_stats = retriever.build_or_load_index()
except Exception as exc:
    index_stats = {"error": str(exc), "total_passages": len(kb.entries), "embedding_dim": 0}


@app.route("/")
def index():
    """Renders the main Smart Campus Assistant web interface."""
    ollama_status = generator.check_ollama_status()
    kb_stats = kb.get_stats()
    categories = kb.get_categories()
    eval_report = evaluator.load_cached_results()
    test_questions = evaluator.load_test_questions()

    return render_template(
        "index.html",
        project=PROJECT_METADATA,
        ollama_status=ollama_status,
        kb_stats=kb_stats,
        categories=categories,
        active_embed_model=retriever.embed_model,
        active_llm_model=generator.llm_model,
        default_top_k=DEFAULT_TOP_K,
        default_threshold=DEFAULT_SIMILARITY_THRESHOLD,
        eval_report=eval_report,
        test_questions=test_questions,
    )


@app.route("/api/status", methods=["GET"])
def api_status():
    """Returns live Ollama service status, installed models, and knowledge base statistics."""
    status = generator.check_ollama_status()
    return jsonify({
        "ollama": status,
        "active_embed_model": retriever.embed_model,
        "active_llm_model": generator.llm_model,
        "kb_stats": kb.get_stats(),
        "project": PROJECT_METADATA,
    })


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """
    Main RAG endpoint:
    1. Cleans and preprocesses the student's natural-language question (NLP)
    2. Creates a dense vector embedding via Ollama
    3. Retrieves Top-K campus passages via Cosine Similarity
    4. Generates a grounded answer via local Ollama LLM (or abstains if unsupported)
    """
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Please enter a valid question."}), 400

    top_k = int(payload.get("top_k", DEFAULT_TOP_K))
    threshold = float(payload.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD))
    requested_llm = payload.get("llm_model") or generator.llm_model
    requested_embed = payload.get("embed_model") or retriever.embed_model

    # Switch embedding model dynamically if user selected a different one
    if requested_embed != retriever.embed_model:
        retriever.embed_model = requested_embed
        retriever.build_or_load_index()

    generator.llm_model = requested_llm

    try:
        retrieval_result = retriever.retrieve(
            question=question,
            top_k=top_k,
            similarity_threshold=threshold,
        )
        response_data = generator.generate_answer(
            question=question,
            retrieval_result=retrieval_result,
            llm_model=requested_llm,
        )
        return jsonify(response_data)
    except Exception as exc:
        return jsonify({
            "error": f"Failed to process question via Ollama: {exc}. Ensure Ollama is running (`ollama serve`)."
        }), 500


@app.route("/api/knowledge-base", methods=["GET"])
def api_get_kb():
    """Returns all verified campus knowledge base passages with optional category/search filter."""
    category = (request.args.get("category") or "").strip()
    search = (request.args.get("search") or "").strip().lower()

    filtered = kb.entries
    if category and category != "All":
        filtered = [e for e in filtered if e.get("category") == category]
    if search:
        filtered = [
            e for e in filtered
            if search in e.get("title", "").lower()
            or search in e.get("content", "").lower()
            or search in e.get("id", "").lower()
            or search in e.get("source", "").lower()
            or any(search in kw.lower() for kw in e.get("keywords", []))
        ]

    return jsonify({
        "entries": filtered,
        "total": len(filtered),
        "categories": kb.get_categories(),
        "stats": kb.get_stats(),
    })


@app.route("/api/knowledge-base", methods=["POST"])
def api_add_kb():
    """Adds a new verified campus passage to the knowledge base and updates the vector index."""
    payload = request.get_json(silent=True) or {}
    title = (payload.get("title") or "").strip()
    category = (payload.get("category") or "").strip()
    content = (payload.get("content") or "").strip()
    source = (payload.get("source") or "").strip()
    keywords_raw = (payload.get("keywords") or "").strip()

    if not title or not category or not content or not source:
        return jsonify({"error": "Title, Category, Content, and Official Source are required."}), 400

    keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()] if keywords_raw else None
    new_entry = kb.add_entry(
        title=title,
        content=content,
        category=category,
        source=source,
        keywords=keywords,
    )
    retriever.entries = kb.entries
    idx_stats = retriever.build_or_load_index()

    return jsonify({
        "message": f"Successfully added {new_entry['id']} and updated vector embeddings.",
        "entry": new_entry,
        "index_stats": idx_stats,
    })


@app.route("/api/evaluation", methods=["GET"])
def api_get_evaluation():
    """Returns cached evaluation benchmark results and the 25 test questions."""
    report = evaluator.load_cached_results()
    return jsonify({
        "report": report,
        "test_questions": evaluator.load_test_questions(),
    })


@app.route("/api/evaluation/run", methods=["POST"])
def api_run_evaluation():
    """Runs the 25-question evaluation benchmark live and returns the report."""
    payload = request.get_json(silent=True) or {}
    top_k = int(payload.get("top_k", DEFAULT_TOP_K))
    threshold = float(payload.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD))

    report = evaluator.run_evaluation(
        top_k=top_k,
        similarity_threshold=threshold,
    )
    return jsonify(report)


if __name__ == "__main__":
    print("=" * 70)
    print(" Smart Campus Assistant (RAG + Ollama) – CSC-320(L) Group 1")
    print(f" Active Embedding Model : {retriever.embed_model}")
    print(f" Active Local LLM Model : {generator.llm_model}")
    print(f" Knowledge Base Entries : {len(kb.entries)} verified passages")
    print(" Server URL             : http://127.0.0.1:5000")
    print("=" * 70)
    app.run(host="127.0.0.1", port=5000, debug=False)
