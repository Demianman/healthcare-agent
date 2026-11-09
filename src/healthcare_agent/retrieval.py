"""Versioned, deterministic hybrid retrieval over a synthetic policy corpus."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass(frozen=True)
class Passage:
    policy_id: str
    version: str
    service: str
    text: str
    source: str

    @property
    def citation(self) -> str:
        return f"{self.policy_id}@{self.version}"


def load_corpus(path: Path) -> list[Passage]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    passages = []
    seen = set()
    for row in raw:
        passage = Passage(**row)
        if not all([passage.policy_id, passage.version, passage.service, passage.text, passage.source]):
            raise ValueError("Policy fields must be nonempty")
        if passage.citation in seen:
            raise ValueError(f"Duplicate policy citation: {passage.citation}")
        seen.add(passage.citation)
        passages.append(passage)
    return passages


class HybridRetriever:
    """BM25 plus character trigram score, with service filtering and reranking."""

    def __init__(self, passages: list[Passage]):
        self.passages = passages
        self.doc_tokens = [tokens(p.text + " " + p.service) for p in passages]
        self.avg_length = sum(map(len, self.doc_tokens)) / max(len(passages), 1)
        self.df = {term: sum(term in doc for doc in self.doc_tokens) for term in set(t for doc in self.doc_tokens for t in doc)}

    @staticmethod
    def _trigrams(text: str) -> set[str]:
        normalized = " ".join(tokens(text))
        return {normalized[i:i + 3] for i in range(max(0, len(normalized) - 2))}

    def search(self, service: str, query: str, k: int = 3) -> list[dict]:
        eligible = [(i, p) for i, p in enumerate(self.passages) if p.service.lower() == service.lower()]
        if not eligible:
            return []
        q_terms = set(tokens(query))
        q_grams = self._trigrams(query)
        scored = []
        for i, passage in eligible:
            doc = self.doc_tokens[i]
            length = len(doc)
            bm25 = 0.0
            for term in q_terms:
                tf = doc.count(term)
                if not tf:
                    continue
                idf = math.log(1 + (len(self.passages) - self.df[term] + 0.5) / (self.df[term] + 0.5))
                bm25 += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / max(self.avg_length, 1)))
            grams = self._trigrams(passage.text + " " + passage.service)
            overlap = len(q_grams & grams) / max(len(q_grams), 1)
            score = bm25 + overlap
            scored.append((score, passage))
        scored.sort(key=lambda item: (-item[0], item[1].citation))
        return [{"id": p.policy_id, "version": p.version, "citation": p.citation,
                 "text": p.text, "source": p.source, "score": round(score, 5)} for score, p in scored[:k]]
