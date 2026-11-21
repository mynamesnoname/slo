"""Knowledge base loading, sectioning, and retrieval.

The KB lives in kb/*.md. We split each document into headed sections and
index them with TF-IDF (sklearn) for lexical retrieval. Retrieval returns
section text plus provenance (file + section), so the agent's use of the
knowledge base is auditable.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

KB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kb")


@dataclass
class KBSection:
    doc: str
    title: str
    text: str

    @property
    def provenance(self):
        return f"kb/{self.doc} :: {self.title}"


def _split_sections(text: str):
    """Split markdown text into sections on '##' or '###' headings."""
    lines = text.splitlines()
    sections = []
    current_title = "top"
    current = []
    for ln in lines:
        if re.match(r"^#{1,3}\s+", ln):
            if current:
                sections.append((current_title, "\n".join(current).strip()))
            current_title = ln.lstrip("#").strip()
            current = []
        else:
            current.append(ln)
    if current:
        sections.append((current_title, "\n".join(current).strip()))
    return [(t, c) for t, c in sections if c]


class KnowledgeBase:
    def __init__(self, kb_dir: str = KB_DIR):
        self.kb_dir = kb_dir
        self.sections: list[KBSection] = []
        self._vectorizer = None
        self._matrix = None
        self._load()

    def _load(self):
        for fn in sorted(os.listdir(self.kb_dir)):
            if not fn.endswith(".md"):
                continue
            path = os.path.join(self.kb_dir, fn)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            for title, body in _split_sections(text):
                self.sections.append(KBSection(doc=fn, title=title, text=body))
        # build index
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2)
        )
        docs = [f"{s.title}\n{s.text}" for s in self.sections]
        self._matrix = self._vectorizer.fit_transform(docs)

    def retrieve(self, query: str, k: int = 4) -> list[tuple[KBSection, float]]:
        """Retrieve top-k sections by TF-IDF cosine similarity."""
        qvec = self._vectorizer.transform([query])
        scores = (self._matrix @ qvec.T).toarray().ravel()
        idx = np.argsort(scores)[::-1][:k]
        return [(self.sections[i], float(scores[i])) for i in idx if scores[i] > 0]

    def retrieve_by_doc(self, doc_prefix: str) -> list[KBSection]:
        """Retrieve all sections from a given doc (e.g. 'lines')."""
        return [s for s in self.sections if s.doc.startswith(doc_prefix)]

    def all_sections_text(self, max_chars: int = 24000) -> str:
        """Full KB text (for system prompt if needed), truncated."""
        parts = []
        total = 0
        for s in self.sections:
            block = f"### [{s.doc}] {s.title}\n{s.text}\n"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n".join(parts)


def build_kb() -> KnowledgeBase:
    return KnowledgeBase()
