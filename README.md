# Smart Campus Assistant using RAG + LLM (Ollama)

**Course:** CSC-320(L) – Artificial Intelligence  
**Section:** Section B | **Group:** Group 1  
**Local AI Stack:** Python 3, Flask, Ollama (`all-minilm` / `nomic-embed-text` + `qwen2.5` / `qwen3`), NumPy (Cosine Similarity)

---

## 1. Group 1 Members & Work Distribution

| S.No | Reg. No. | Student Name | Role & Implemented Modules |
| :--- | :--- | :--- | :--- |
| **1** | **18352** | **Abdullah** | **Module 1 (Lead AI & Core RAG Retrieval Engine):** Dense vector embeddings via Ollama (`/api/embed`), NumPy Cosine Similarity calculation, hybrid scoring, similarity threshold & out-of-scope abstention guard (`src/retriever.py`, `src/config.py`) |
| **2** | **18333** | **Aftab** | **Module 2 (NLP Preprocessing & Local LLM Generation):** NLP question cleaning, abbreviation expansion, document chunking & Ollama LLM API integration (`/api/chat`) for grounded answer generation (`src/nlp_processor.py`, `src/llm_generator.py`) |
| **3** | **18365** | **Talha** | **Module 3 (Flask Web Application & Frontend UI):** Flask backend routes, REST APIs & interactive 4-tab Live RAG Inspector interface (`app.py`, `templates/index.html`, `static/js/app.js`, `static/css/style.css`) |
| **4** | **18292** | **Shahzad** | **Module 4 (Evaluation Benchmark & Response Testing):** Automated 25-question test benchmark, Retrieval Hit@1/Hit@3, answer accuracy, abstention testing & latency profiling (`src/evaluator.py`, `evaluate.py`, `data/test_questions.json`) |
| **5** | **18165** | **Hafeez Khan** | **Module 5 (Campus Knowledge Base & Documentation):** Collecting, verifying & organizing 38 official campus FAQ entries and documents (JSON/CSV/TXT) with sources, technical report & presentation (`data/campus_knowledge_base.json`, `src/knowledge_base.py`, `docs/TECHNICAL_REPORT.md`) |

---

## 2. Project Overview & Objectives Achieved

University students frequently need accurate, up-to-date answers regarding admission procedures, tuition fees, scholarships, department locations, office timings, library rules, attendance policies (75% rule), and IT/LMS support.

This prototype implements a complete **Retrieval-Augmented Generation (RAG)** workflow running **100% locally via Ollama**:
1. **Verified Campus Knowledge Base (Objective 1):** Contains **38 verified campus passages** (36 structured FAQ entries in `data/campus_knowledge_base.json` & `.csv` + 2 chunked sections from `data/documents/campus_notices_supplement.txt`), exceeding the minimum requirement of 30 entries. Every passage records its official university source.
2. **NLP Question Preprocessing & Embeddings (Objective 2):** Cleans queries, expands contractions and university abbreviations (`BSCS`, `BSAI`, `LMS`, `HoD`, `FYP`, `CGPA`, `UAT`), extracts domain tokens, and generates dense vector embeddings locally via Ollama (`/api/embed`).
3. **Semantic Retrieval via Cosine Similarity (Objective 3):** Computes vector Cosine Similarity in NumPy along with lexical keyword overlap to retrieve the Top-K most relevant campus passages and detect out-of-scope queries.
4. **Grounded Local LLM Generation & Abstention (Objective 4):** Connects retrieved passages to a local Ollama language model (`/api/chat`) using a strict context-grounded system prompt that cites official sources (`[KB-xxx]`) and **abstains cleanly** when the knowledge base does not contain sufficient information.
5. **25-Question Prototype Evaluation (Objective 5):** Evaluates 25 test questions (19 in-scope campus questions + 6 out-of-scope abstention tests), recording Retrieval Hit@1, Hit@3, Answer Accuracy, Abstention Accuracy, and Response Latency (`data/evaluation_results.json` & `.csv`).

---

## 3. Project Directory Structure

```text
smart-campus-assistant/
├── app.py                              # Flask web server & REST API endpoints (Abdullah - 18352)
├── evaluate.py                         # CLI script to run the 25-question benchmark (Hafeez Khan - 18165)
├── requirements.txt                    # Python package dependencies
├── README.md                           # Project setup & usage guide
├── docs/
│   └── TECHNICAL_REPORT.md             # Full Technical Report, Architecture & Viva Q&A Guide
├── data/
│   ├── campus_knowledge_base.json      # 36 structured verified campus entries with sources (Talha - 18365)
│   ├── campus_knowledge_base.csv       # Auto-exported CSV version of the knowledge base
│   ├── embeddings_cache.json           # Local cache of Ollama dense vector embeddings
│   ├── test_questions.json             # 25 benchmark test questions (19 in-scope + 6 abstention)
│   ├── evaluation_results.json         # Detailed JSON evaluation output
│   ├── evaluation_results.csv          # Tabular CSV evaluation output
│   └── documents/
│       └── campus_notices_supplement.txt  # Raw text document split into passages automatically
├── src/
│   ├── config.py                       # Configuration, thresholds, paths & Group 1 metadata
│   ├── knowledge_base.py               # Knowledge base loader, chunker & CSV exporter (Talha - 18365)
│   ├── nlp_processor.py                # NLP cleaning, abbreviation expansion & tokenization (Aftab - 18333)
│   ├── retriever.py                    # Ollama embeddings & Cosine Similarity retrieval (Aftab & Shahzad)
│   ├── llm_generator.py                # Ollama LLM chat integration & abstention guard (Shahzad - 18292)
│   └── evaluator.py                    # Benchmark evaluation engine (Hafeez Khan - 18165)
├── templates/
│   └── index.html                      # Interactive 4-tab web dashboard & Live RAG Inspector
└── static/
    ├── css/style.css                   # Responsive dark-mode academic UI stylesheet
    └── js/app.js                       # Frontend logic for chat, inspector, KB explorer & benchmark
```

---

## 4. Quick Start & Running the Project

### Step 1: Verify Ollama is Running Locally
Make sure Ollama is running and has an embedding model and a language model pulled:
```powershell
ollama list
# Already installed on this system:
# - all-minilm:latest (Embedding model)
# - qwen2.5:0.5b      (Local LLM)

# Optional: You can also pull nomic-embed-text or qwen3 if desired:
# ollama pull nomic-embed-text
# ollama pull qwen3:0.6b
```

### Step 2: Run the 25-Question Evaluation Benchmark (CLI)
```powershell
.\.venv\Scripts\python.exe evaluate.py
```

### Step 3: Start the Flask Web Application
```powershell
.\.venv\Scripts\python.exe app.py
```
Then open **http://127.0.0.1:5000** in your browser.
