# Technical Report: Smart Campus Assistant using RAG + LLM (Ollama)

**Course:** CSC-320(L) – Artificial Intelligence  
**Section:** Section B | **Group Number:** Group 1  
**Allocated Project Title:** Smart Campus Assistant using RAG + LLM  

---

## 1. Group Members & Contributions

| S.No | Reg. No. | Student Name | Role & Technical Responsibility |
| :--- | :--- | :--- | :--- |
| **1** | **18352** | **Abdullah** | Group coordination, requirements gathering, Flask web application & system integration (`app.py`, `templates/index.html`, `static/`) |
| **2** | **18365** | **Talha** | Collection, verification & structuring of 38 official campus information passages (`data/campus_knowledge_base.json`, `data/campus_knowledge_base.csv`, `src/knowledge_base.py`) |
| **3** | **18333** | **Aftab** | NLP question preprocessing, abbreviation expansion, document chunking & Cosine Similarity retrieval (`src/nlp_processor.py`, `src/retriever.py`) |
| **4** | **18292** | **Shahzad** | Local Ollama configuration, embedding & LLM REST API integration, grounded prompt engineering & abstention guard (`src/llm_generator.py`) |
| **5** | **18165** | **Hafeez Khan** | 25-question benchmark design, retrieval/generation/abstention evaluation, metrics logging & technical documentation (`src/evaluator.py`, `evaluate.py`) |

---

## 2. Problem Statement & Motivation

University students regularly need reliable answers regarding routine academic and administrative services—including admission merit formulas, BSCS/BSAI eligibility criteria, semester tuition fees, scholarship policies, department and laboratory locations, office hours, library borrowing rules, the mandatory 75% attendance regulation, and LMS/Wi-Fi troubleshooting.

In practice, this information is scattered across multiple PDF prospectuses, notice-board circulars, and academic handbooks. Students either spend substantial time searching or rely on informal word-of-mouth that can be inaccurate or outdated. Standard standalone Large Language Models (LLMs) cannot solve this problem directly because they do not know university-specific rules and may hallucinate plausible-sounding but incorrect policies.

To address this problem, Group 1 developed the **Smart Campus Assistant using Retrieval-Augmented Generation (RAG) + Local LLM (Ollama)**. The system:
1. Accepts student questions in natural language via a Flask web interface.
2. Applies NLP preprocessing and converts both campus documents and student questions into dense vector embeddings using a local pretrained embedding model in Ollama.
3. Retrieves the most relevant verified campus passages using **Cosine Similarity**.
4. Supplies those passages as grounded context to a locally running Ollama LLM to generate an accurate, source-cited answer—or explicitly **abstains** when the knowledge base does not contain the answer.

---

## 3. Data Sources & Knowledge Base Organization (Objective 1)

Prepared and verified by **Talha (18365)** and **Abdullah (18352)**, the knowledge base comprises **38 verified campus information entries** (exceeding the proposal objective of at least 30 passages) organized across **10 categories**:

1. **Admissions & Registration (`KB-001` – `KB-004`):** Undergraduate admission procedure, UAT merit calculation formula (50% Intermediate, 40% Test, 10% Matric), BSCS/BSAI eligibility (including Pre-Medical deficiency math rules), course registration & add/drop/withdrawal deadlines, and Admission Office contact details.
2. **Fee Information & Scholarships (`KB-005` – `KB-008`):** Per-credit-hour fee (Rs. 4,500/credit hour; Rs. 89,500 regular 18-credit semester total), challan payment and Rs. 200/day late fine policy, Merit Scholarships (75%, 50%, 25% waivers for SGPA ≥ 3.75), Need-Based Aid, 25% Kinship/Sibling discount, and HEC tuition refund policy.
3. **Departments & Locations (`KB-009` – `KB-013`):** Block C (CS & AI Department, HoD Office C-101, AI Lab 4), Block A (Registrar A-102, Exam Cell A-112, One-Window G-01), Block B & D locations, Campus Medical Center (`SC-02`, Emergency Ext. `1122`), and Cafeteria/Mosque locations.
4. **Office Timings & Student Services (`KB-014` – `KB-017`):** Regular administrative hours (8:30 AM – 4:30 PM, public dealing 9:00 AM – 3:30 PM), Friday & Ramadan schedules, faculty consultation hours, and Bonafide/Transcript issuance fees and turnaround times.
5. **Library Rules & Resources (`KB-018` – `KB-021`):** Central Library timings (8:00 AM – 7:00 PM weekdays, 9:00 AM – 3:00 PM Saturdays), borrowing limits (4 books for 14 days for undergraduates), overdue fines (Rs. 20/day), and HEC Digital Library / IEEE / Turnitin (<19% similarity) access.
6. **Examinations & Academic Policy (`KB-022` – `KB-026`):** Mandatory 75% attendance rule, course weightage breakdown (Midterm 25%, Final 50%, Quizzes/Assignments 25%), 4.00 GPA grading scale, academic probation (<2.00 CGPA), and make-up examination policy.
7. **IT Support, LMS & Wi-Fi (`KB-027` – `KB-029`):** LMS portal login & password reset at IT Helpdesk (`C-002`), Campus Wi-Fi & `eduroam` MAC registration, Microsoft 365 student email, and AI/GPU Lab rules.
8. **Hostel, Transport & Facilities (`KB-030` – `KB-033`):** Boys/Girls hostel allotment, semester charges (Rs. 36,000) & curfew times (10:00 PM male, 8:30 PM female), 12 city bus routes (Rs. 22,000/semester, departure at 1:45 PM & 4:35 PM), Sports Complex hours, duplicate ID card (Rs. 800), and vehicle parking stickers.
9. **Student Support & Career Services (`KB-034` – `KB-036`):** Mandatory 6-to-8-week summer internship after 6th semester, FYP-I & FYP-II registration and defense rules, student societies, psychological counseling (`SC-06`), and Anti-Harassment Cell (`A-210`).
10. **Academic & Examination Notices (`KB-037` – `KB-038`):** Automatically chunked from raw text (`data/documents/campus_notices_supplement.txt`) covering Convocation clearance and Summer Semester rules.

---

## 4. System Architecture & Methodology (Objectives 2, 3 & 4)

```
+-------------------------------------------------------------------------+
|                        STUDENT WEB INTERFACE (Flask)                    |
|              Natural-Language Question (e.g., "What is BSAI fee?")      |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| 1. NLP QUESTION PREPROCESSING (src/nlp_processor.py - Aftab 18333)      |
|  - Unicode & whitespace normalization, contraction expansion            |
|  - Campus abbreviation expansion (BSAI -> BS Artificial Intelligence)   |
|  - Stopword removal, keyword extraction & category intent detection     |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| 2. DENSE VECTOR EMBEDDING (src/retriever.py - Ollama /api/embed)        |
|  - Pretrained embedding model (all-minilm:latest / nomic-embed-text)    |
|  - Converts query & 38 KB passages into normalized dense vectors        |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| 3. SEMANTIC RETRIEVAL & ABSTENTION CHECK (src/retriever.py)             |
|  - Computes Cosine Similarity: cos(theta) = (q . d) / (||q|| * ||d||)   |
|  - Combines dense semantic score (80%) + specific keyword overlap (20%) |
|  - Checks relevance threshold & vocabulary coverage to filter out-of-   |
|    scope questions before LLM generation                                |
+------------------------------------+------------------------------------+
                                     |
                      +--------------+--------------+
                      |                             |
             [Relevant Context]            [Below Threshold / Out-of-Scope]
                      |                             |
                      v                             v
+------------------------------------+   +--------------------------------+
| 4. GROUNDED OLLAMA LLM GENERATION  |   | 5. SAFE ABSTENTION RESPONSE    |
|  (src/llm_generator.py - Shahzad)  |   |  Clearly states that the KB    |
|  - Local LLM (qwen2.5 / qwen3)     |   |  does not contain verified     |
|  - Answers strictly from Top-K     |   |  information on that topic     |
|    passages & cites [KB-xxx]       |   |  (Zero Hallucination)          |
+------------------------------------+   +--------------------------------+
```

### Mathematical Formulation of Retrieval
For a student query vector $\mathbf{q} \in \mathbb{R}^d$ and each knowledge base passage vector $\mathbf{d}_i \in \mathbb{R}^d$ produced by the local Ollama embedding model, the semantic similarity is calculated via **Cosine Similarity**:

$$\text{CosineSimilarity}(\mathbf{q}, \mathbf{d}_i) = \frac{\mathbf{q} \cdot \mathbf{d}_i}{\|\mathbf{q}\|_2 \, \|\mathbf{d}_i\|_2}$$

To ensure exact campus identifiers (such as room numbers `G-04`, `C-101`, percentages `75%`, and degree codes `BSCS`, `BSAI`) receive boosted precision while out-of-domain queries are cleanly separated, the retriever computes a hybrid score:

$$\text{Score}(q, d_i) = 0.80 \times \text{CosineSimilarity}(\mathbf{q}, \mathbf{d}_i) + 0.20 \times \text{KeywordOverlap}(q, d_i)$$

---

## 5. Limitations & Future Improvements

1. **Current Limitations:**
   - **Text-Only Scope:** In accordance with the allocated project title (*Smart Campus Assistant using RAG + LLM*), the system processes text documents and FAQs only; scanned image notices without OCR or computer vision are outside the scope.
   - **Local Hardware Latency:** Running LLM inference on CPU via Ollama takes ~1–4 seconds per response depending on answer length and model size (`qwen2.5:0.5b` vs. larger 3B/7B models).
   - **Static Snapshot of Notices:** While new entries can be added via the web UI, real-time synchronization with the live university LMS database is not yet connected.

2. **Future Improvements:**
   - Add PDF/OCR ingestion for scanned notification letters.
   - Support multi-turn conversational follow-up memory across a chat session.
   - Integrate bilingual Urdu / Roman-Urdu query translation for broader student accessibility.

---

## 6. Viva Preparation Questions & Answers (For Group 1 Members)

### Q1 (For Abdullah – 18352): How does the Flask application integrate the complete RAG workflow?
- **Answer:** When a student submits a question on the web interface, the frontend sends a `POST /api/ask` JSON request to `app.py`. The Flask endpoint invokes `retriever.retrieve(question)` (which runs NLP preprocessing in `nlp_processor.py`, fetches the query embedding from Ollama's `/api/embed` API, and ranks the 38 campus passages using cosine similarity). Next, Flask passes the retrieval result to `generator.generate_answer()` in `llm_generator.py`, which either triggers the abstention guard (if relevance is below threshold) or sends the Top-K passages to Ollama's `/api/chat` endpoint to synthesize a grounded response with source citations.

### Q2 (For Talha – 18365): How was the campus knowledge base prepared and why is source tracking important in RAG?
- **Answer:** We curated 38 verified campus information passages across 10 university categories (Admissions, Fees, Scholarships, Locations, Timings, Library, Exams, IT/LMS, Hostels/Transport, and Career Services) in `data/campus_knowledge_base.json` and `campus_knowledge_base.csv`, plus raw text notices in `data/documents/`. Each entry stores a unique ID (`KB-001` to `KB-038`), category, title, verified passage text, keywords, and the exact official source (e.g., *Official Admission Prospectus 2026-27, Section 2.1*). Recording sources allows the assistant to cite where every fact came from so students can verify policies and avoid misinformation.

### Q3 (For Aftab – 18333): What NLP preprocessing steps are performed before creating embeddings, and how does Cosine Similarity work?
- **Answer:** In `src/nlp_processor.py`, we perform Unicode normalization, contraction expansion, punctuation cleanup, stopword removal, and **campus abbreviation expansion** (e.g., expanding `BSAI` to `BS Artificial Intelligence`, `LMS` to `Learning Management System portal`, `HoD` to `Head of Department`). Then we convert the text into dense numerical vectors using a pretrained embedding model in Ollama (`all-minilm:latest` / `nomic-embed-text`). We normalize each vector to unit length ($\|\mathbf{v}\|_2 = 1$) in NumPy and compute the dot product $\mathbf{q} \cdot \mathbf{d}_i$, which equals the cosine of the angle between the question vector and each passage vector. Values closer to `1.0` indicate high semantic similarity.

### Q4 (For Shahzad – 18292): How does the system communicate with Ollama locally, and how do you prevent LLM hallucination?
- **Answer:** Ollama runs a local HTTP server on `http://localhost:11434`. In `src/llm_generator.py`, we use Python's `requests` library to call `/api/embed` for vector embeddings and `/api/chat` for text generation with `qwen2.5:0.5b` (or `qwen3`). We prevent hallucination in two ways:
  1. **Pre-LLM Abstention Guard:** If the retrieved passages fall below the similarity threshold or have zero campus topic vocabulary coverage, we block LLM generation and return a verified abstention message immediately.
  2. **Strict Grounded Prompting & Low Temperature:** We set `temperature = 0.1` and instruct the model via the system prompt to use *only* the provided passages, cite `[KB-xxx]`, and output `INSUFFICIENT_CONTEXT` if the passages do not answer the question.

### Q5 (For Hafeez Khan – 18165): How did you evaluate the prototype and what metrics did you record?
- **Answer:** In `src/evaluator.py` and `data/test_questions.json`, we created a benchmark of **25 test questions**: 19 in-scope campus questions mapped to ground-truth `KB` passage IDs and expected factual keywords, plus 6 out-of-scope questions (cooking recipes, sports trivia, stock prices, medical prescriptions, visa rules, and nonexistent campus airports) where the system *must* abstain. We recorded **Retrieval Hit@1**, **Retrieval Hit@3**, **In-Scope Answer Accuracy**, **Out-of-Scope Abstention Accuracy**, and **Response Time (ms)** across embedding, retrieval, and LLM generation.
