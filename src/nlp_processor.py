"""
NLP Preprocessing and Document Chunking Module
Author / Role: Aftab (Reg No. 18333) – NLP question preparation, text cleaning & document splitting.

Implements:
1. Text cleaning and normalization (Unicode cleanup, whitespace normalization)
2. Campus abbreviation & contraction expansion (e.g., HOD -> Head of Department, LMS, FYP, BSCS, BSAI)
3. Tokenization and stopword removal for keyword extraction
4. Query category / intent detection
5. Document passage splitting (chunking raw text files into searchable passages)
"""

import re
import unicodedata
from typing import Dict, List, Any

# Standard English stopwords (excluding key domain modifiers like numbers/units)
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "i", "me", "my",
    "we", "our", "you", "your", "he", "she", "it", "they", "them", "their",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "if", "or", "because", "as", "until", "while", "of", "at", "by",
    "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "please", "tell", "know", "want", "need", "get", "give",
}

# English contractions mapping
CONTRACTIONS = {
    "what's": "what is",
    "where's": "where is",
    "when's": "when is",
    "who's": "who is",
    "how's": "how is",
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "i'm": "i am",
    "i've": "i have",
    "i'd": "i would",
    "it's": "it is",
}

# University & Campus domain abbreviation expansions
CAMPUS_ABBREVIATIONS = {
    "bscs": "bs computer science bscs",
    "bsai": "bs artificial intelligence bsai",
    "bsse": "bs software engineering bsse",
    "cs": "computer science cs",
    "ai": "artificial intelligence ai",
    "hod": "head of department hod",
    "lms": "learning management system portal lms",
    "cms": "campus management system portal cms",
    "fyp": "final year project fyp",
    "cgpa": "cumulative grade point average cgpa",
    "gpa": "grade point average gpa",
    "sgpa": "semester grade point average sgpa",
    "uat": "university admission test uat",
    "hec": "higher education commission hec",
    "dept": "department",
    "wifi": "wi-fi wireless internet wifi",
    "id": "identification card id",
    "opac": "online library catalog opac",
    "sa": "short of attendance",
}

# Category keyword mapping for query intent detection
CATEGORY_INTENT_KEYWORDS = {
    "Admissions & Registration": {
        "admission", "apply", "merit", "eligibility", "uat", "entry", "test",
        "register", "registration", "add", "drop", "withdraw", "pre-medical", "fsc", "ics"
    },
    "Fee Information & Scholarships": {
        "fee", "tuition", "challan", "voucher", "scholarship", "financial", "aid",
        "refund", "fine", "installment", "credit", "cost", "charges", "sibling", "kinship", "waiver"
    },
    "Departments & Locations": {
        "where", "location", "located", "block", "room", "floor", "department",
        "office", "hod", "registrar", "medical", "dispensary", "ambulance", "cafeteria", "mosque"
    },
    "Office Timings & Student Services": {
        "timing", "timings", "hours", "open", "close", "break", "friday", "ramadan",
        "consultation", "advisor", "bonafide", "transcript", "certificate", "one-window"
    },
    "Library Rules & Resources": {
        "library", "book", "books", "borrow", "loan", "overdue", "renew",
        "turnitin", "plagiarism", "ieee", "digital", "reading"
    },
    "Examinations & Academic Policy": {
        "exam", "examination", "midterm", "final", "attendance", "75%", "grading",
        "grade", "gpa", "cgpa", "probation", "dismissal", "makeup", "retake", "quiz", "weightage"
    },
    "IT Support, LMS & Wi-Fi": {
        "lms", "portal", "password", "login", "wifi", "wi-fi", "internet",
        "eduroam", "mac", "email", "helpdesk", "lab", "gpu"
    },
    "Hostel, Transport & Facilities": {
        "hostel", "accommodation", "mess", "curfew", "warden", "bus", "transport",
        "shuttle", "route", "sports", "gym", "parking", "sticker", "card"
    },
    "Student Support & Career Services": {
        "internship", "career", "placement", "job", "fyp", "project", "society",
        "societies", "club", "counseling", "harassment", "complaint"
    },
}


def clean_text(text: str) -> str:
    """
    Cleans raw text by normalizing Unicode characters, expanding contractions,
    removing noisy control symbols, and collapsing multiple spaces.
    """
    if not text:
        return ""
    # Normalize unicode characters
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()

    # Expand common contractions (case-insensitive)
    lower_text = text.lower()
    for contraction, expansion in CONTRACTIONS.items():
        lower_text = re.sub(r"\b" + re.escape(contraction) + r"\b", expansion, lower_text)

    # Keep alphanumeric, spaces, percentages, hyphens, dots, @, and currency/numbers
    cleaned = re.sub(r"[^\w\s%\-\.@/,:()]", " ", lower_text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def expand_campus_abbreviations(text: str) -> tuple[str, List[str]]:
    """
    Expands university-specific abbreviations so embedding models and keyword
    matching capture full semantic meaning.
    Returns (expanded_text, list_of_expanded_terms).
    """
    words = text.split()
    expanded_words = []
    expansions_applied = []

    for word in words:
        clean_w = re.sub(r"[^\w\-]", "", word.lower())
        if clean_w in CAMPUS_ABBREVIATIONS:
            expansion = CAMPUS_ABBREVIATIONS[clean_w]
            expanded_words.append(expansion)
            expansions_applied.append(f"{clean_w.upper()} → {expansion}")
        else:
            expanded_words.append(word)

    return " ".join(expanded_words), expansions_applied


def tokenize_and_extract_keywords(text: str) -> List[str]:
    """
    Tokenizes cleaned text and removes stopwords to extract informative keywords.
    Preserves room numbers (e.g. G-04, C-101), percentages (75%), and numbers.
    """
    raw_tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*%?", text.lower())
    keywords = [
        tok for tok in raw_tokens
        if tok not in STOPWORDS and (len(tok) > 1 or tok.isdigit())
    ]
    # Deduplicate while preserving order
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)
    return unique_keywords


def detect_query_intent(keywords: List[str]) -> Dict[str, Any]:
    """
    Predicts the most likely campus category for a student's question based on
    extracted keywords and checks whether any domain terms are present.
    """
    keyword_set = set(keywords)
    category_scores = {}

    for category, intent_words in CATEGORY_INTENT_KEYWORDS.items():
        overlap = keyword_set.intersection(intent_words)
        if overlap:
            category_scores[category] = len(overlap)

    if not category_scores:
        return {
            "predicted_category": "Unclassified / General",
            "matched_domain_terms": [],
            "has_campus_domain_signal": False,
        }

    best_category = max(category_scores, key=category_scores.get)
    all_domain_words = set().union(*CATEGORY_INTENT_KEYWORDS.values())
    matched_terms = sorted(list(keyword_set.intersection(all_domain_words)))

    return {
        "predicted_category": best_category,
        "matched_domain_terms": matched_terms,
        "has_campus_domain_signal": True,
    }


def preprocess_question(question: str) -> Dict[str, Any]:
    """
    Complete NLP preprocessing pipeline for a student's natural-language question.
    Returns structured metadata for both embedding generation and UI inspection.
    """
    cleaned = clean_text(question)
    expanded_text, expansions = expand_campus_abbreviations(cleaned)
    keywords = tokenize_and_extract_keywords(cleaned)
    expanded_keywords = tokenize_and_extract_keywords(expanded_text)
    intent_info = detect_query_intent(expanded_keywords)

    return {
        "original_question": question.strip(),
        "cleaned_question": cleaned,
        "expanded_question": expanded_text,
        "keywords": keywords,
        "expanded_keywords": expanded_keywords,
        "abbreviations_expanded": expansions,
        "predicted_category": intent_info["predicted_category"],
        "matched_domain_terms": intent_info["matched_domain_terms"],
        "has_campus_domain_signal": intent_info["has_campus_domain_signal"],
    }


def format_passage_for_embedding(entry: Dict[str, Any]) -> str:
    """
    Formats a knowledge base entry into a rich semantic text representation
    suitable for dense embedding generation.
    """
    category = entry.get("category", "")
    title = entry.get("title", "")
    content = entry.get("content", "")
    keywords = ", ".join(entry.get("keywords", []))
    return f"Category: {category} | Topic: {title} | Information: {content} | Keywords: {keywords}"


def chunk_document_text(
    raw_text: str,
    source_name: str,
    category: str = "Official Campus Document",
    max_words: int = 120,
    overlap_words: int = 20,
) -> List[Dict[str, Any]]:
    """
    Splits a raw text document into searchable passages (chunks) with source metadata.
    Supports paragraph/section boundaries first, then word-window chunking if a paragraph is long.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]
    passages = []
    chunk_idx = 1

    for para in paragraphs:
        lines = [line.strip() for line in para.splitlines() if line.strip()]
        if not lines:
            continue

        # Check if first line looks like a heading
        if len(lines) > 1 and len(lines[0].split()) <= 12 and not lines[0].endswith("."):
            title = lines[0].lstrip("#*- ").strip()
            body = " ".join(lines[1:])
        else:
            title = f"{source_name} – Section {chunk_idx}"
            body = " ".join(lines)

        words = body.split()
        if len(words) <= max_words:
            kws = tokenize_and_extract_keywords(f"{title} {body}")[:10]
            passages.append({
                "id": f"DOC-{chunk_idx:03d}",
                "category": category,
                "title": title,
                "content": body,
                "source": source_name,
                "keywords": kws,
            })
            chunk_idx += 1
        else:
            step = max(1, max_words - overlap_words)
            for start in range(0, len(words), step):
                slice_words = words[start : start + max_words]
                if len(slice_words) < 15 and passages:
                    break
                chunk_body = " ".join(slice_words)
                kws = tokenize_and_extract_keywords(f"{title} {chunk_body}")[:10]
                passages.append({
                    "id": f"DOC-{chunk_idx:03d}",
                    "category": category,
                    "title": f"{title} (Part {start // step + 1})",
                    "content": chunk_body,
                    "source": source_name,
                    "keywords": kws,
                })
                chunk_idx += 1

    return passages
