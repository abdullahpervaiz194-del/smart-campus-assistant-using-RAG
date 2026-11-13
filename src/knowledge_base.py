"""
Campus Knowledge Base Management Module
Author / Role: Talha (Reg No. 18365) – Collecting, verifying and organizing campus information.

Manages loading, validating, searching, exporting (JSON/CSV), and extending
the verified campus information passages and raw text documents.
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import KB_JSON_PATH, KB_CSV_PATH, DOCS_DIR
from src.nlp_processor import chunk_document_text, tokenize_and_extract_keywords


class CampusKnowledgeBase:
    """
    Loads and manages verified campus information entries from JSON, CSV, and
    supplementary TXT documents in data/documents/.
    """

    def __init__(self, json_path: Path = KB_JSON_PATH, docs_dir: Path = DOCS_DIR):
        self.json_path = json_path
        self.docs_dir = docs_dir
        self.entries: List[Dict[str, Any]] = []
        self.load()

    def load(self) -> List[Dict[str, Any]]:
        """Loads verified entries from JSON and any supplementary .txt files in data/documents/."""
        loaded_entries: List[Dict[str, Any]] = []

        if self.json_path.exists():
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    loaded_entries.extend(data)

        # Also load and chunk any supplementary .txt documents if not already in JSON
        existing_ids = {item["id"] for item in loaded_entries}
        existing_titles = {item["title"].lower() for item in loaded_entries}

        if self.docs_dir.exists():
            for txt_file in sorted(self.docs_dir.glob("*.txt")):
                raw_text = txt_file.read_text(encoding="utf-8")
                chunks = chunk_document_text(
                    raw_text=raw_text,
                    source_name=f"Supplementary Document ({txt_file.name})",
                    category="Academic & Examination Notices",
                )
                for chunk in chunks:
                    if chunk["title"].lower() not in existing_titles:
                        new_id = f"KB-{len(loaded_entries) + 1:03d}"
                        while new_id in existing_ids:
                            new_id = f"KB-{int(new_id.split('-')[1]) + 1:03d}"
                        chunk["id"] = new_id
                        existing_ids.add(new_id)
                        existing_titles.add(chunk["title"].lower())
                        loaded_entries.append(chunk)

        self.entries = loaded_entries
        self.export_to_csv(KB_CSV_PATH)
        return self.entries

    def export_to_csv(self, csv_path: Path = KB_CSV_PATH) -> None:
        """Exports the current knowledge base to CSV format for inspection and report submission."""
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["id", "category", "title", "content", "source", "keywords"]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self.entries:
                writer.writerow({
                    "id": entry.get("id", ""),
                    "category": entry.get("category", ""),
                    "title": entry.get("title", ""),
                    "content": entry.get("content", ""),
                    "source": entry.get("source", ""),
                    "keywords": ", ".join(entry.get("keywords", [])),
                })

    def add_entry(
        self,
        title: str,
        content: str,
        category: str,
        source: str,
        keywords: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Adds a new verified campus entry to the JSON knowledge base and updates CSV."""
        next_num = len(self.entries) + 1
        existing_ids = {e["id"] for e in self.entries}
        new_id = f"KB-{next_num:03d}"
        while new_id in existing_ids:
            next_num += 1
            new_id = f"KB-{next_num:03d}"

        if not keywords:
            keywords = tokenize_and_extract_keywords(f"{title} {content}")[:10]

        new_entry = {
            "id": new_id,
            "category": category.strip(),
            "title": title.strip(),
            "content": content.strip(),
            "source": source.strip(),
            "keywords": keywords,
        }

        # Persist to JSON
        json_entries = []
        if self.json_path.exists():
            with open(self.json_path, "r", encoding="utf-8") as f:
                json_entries = json.load(f)
        json_entries.append(new_entry)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(json_entries, f, indent=2, ensure_ascii=False)

        self.entries.append(new_entry)
        self.export_to_csv(KB_CSV_PATH)
        return new_entry

    def update_entry(
        self,
        entry_id: str,
        title: str,
        content: str,
        category: str,
        source: str,
        keywords: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Updates an existing verified campus entry in JSON, memory, and CSV."""
        if not keywords:
            keywords = tokenize_and_extract_keywords(f"{title} {content}")[:12]

        updated_entry = None
        for entry in self.entries:
            if entry.get("id") == entry_id:
                entry["title"] = title.strip()
                entry["content"] = content.strip()
                entry["category"] = category.strip()
                entry["source"] = source.strip()
                entry["keywords"] = keywords
                updated_entry = entry
                break

        if not updated_entry:
            return None

        # Persist to JSON if it is one of the JSON entries
        if self.json_path.exists():
            with open(self.json_path, "r", encoding="utf-8") as f:
                json_entries = json.load(f)
            found_in_json = False
            for je in json_entries:
                if je.get("id") == entry_id:
                    je.update(updated_entry)
                    found_in_json = True
                    break
            if not found_in_json:
                json_entries.append(updated_entry)
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(json_entries, f, indent=2, ensure_ascii=False)

        self.export_to_csv(KB_CSV_PATH)
        return updated_entry

    def get_categories(self) -> List[str]:
        """Returns sorted list of unique categories in the knowledge base."""
        return sorted(list({e.get("category", "General") for e in self.entries}))

    def get_stats(self) -> Dict[str, Any]:
        """Returns summary statistics of the knowledge base."""
        categories = {}
        for e in self.entries:
            cat = e.get("category", "General")
            categories[cat] = categories.get(cat, 0) + 1
        return {
            "total_entries": len(self.entries),
            "total_categories": len(categories),
            "categories_breakdown": categories,
        }
