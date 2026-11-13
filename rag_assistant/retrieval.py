"""Small, dependency-free retrieval over the curated PawPal+ knowledge base."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "can", "could",
    "do", "does", "for", "from", "had", "has", "have", "how", "i", "in",
    "is", "it", "me", "my", "often",
    "of", "on", "or", "should", "tell", "that", "the", "their", "them",
    "there", "this", "to", "what", "when", "where", "which", "who", "why",
    "with", "you", "your", "pet", "pets", "animal", "animals", "care",
}


@dataclass(frozen=True)
class KnowledgeChunk:
    """One sourced passage that can be retrieved and cited in an answer."""

    id: str
    title: str
    content: str
    source_name: str
    source_url: str
    keywords: tuple[str, ...]


def _normalize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def tokenize(text: str) -> set[str]:
    """Return normalized, meaningful terms for retrieval and evaluation."""
    return {
        _normalize_token(token)
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOP_WORDS and len(token) > 1
    }


class KnowledgeRetriever:
    """Rank local knowledge passages by meaningful query-term overlap."""

    def __init__(
        self,
        knowledge_path: str | Path | None = None,
        *,
        top_k: int = 3,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        path = Path(knowledge_path) if knowledge_path else (
            Path(__file__).resolve().parent.parent
            / "knowledge_base"
            / "pet_care.json"
        )
        try:
            with path.open(encoding="utf-8") as knowledge_file:
                records: Any = json.load(knowledge_file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("The pet-care knowledge base could not be loaded.") from error

        if not isinstance(records, list) or not records:
            raise ValueError("The pet-care knowledge base must contain documents.")

        self.chunks = tuple(self._parse_record(record) for record in records)
        self.top_k = top_k

    @staticmethod
    def _parse_record(record: Any) -> KnowledgeChunk:
        required = ("id", "title", "content", "source_name", "source_url")
        if not isinstance(record, dict) or any(
            not isinstance(record.get(key), str) or not record[key].strip()
            for key in required
        ):
            raise ValueError("A knowledge-base document is missing required text fields.")
        keywords = record.get("keywords", [])
        if not isinstance(keywords, list) or not all(
            isinstance(keyword, str) for keyword in keywords
        ):
            raise ValueError("Knowledge-base keywords must be a list of strings.")
        return KnowledgeChunk(
            id=record["id"],
            title=record["title"],
            content=record["content"],
            source_name=record["source_name"],
            source_url=record["source_url"],
            keywords=tuple(keywords),
        )

    def retrieve(self, query: str) -> list[KnowledgeChunk]:
        """Return relevant passages, or an empty list when no terms match."""
        if not isinstance(query, str):
            raise ValueError("Question must be text.")
        query_terms = tokenize(query)
        if not query_terms:
            return []

        scored: list[tuple[int, int, KnowledgeChunk]] = []
        for index, chunk in enumerate(self.chunks):
            document_terms = tokenize(
                f"{chunk.title} {chunk.content} {' '.join(chunk.keywords)}"
            )
            score = len(query_terms & document_terms)
            if score:
                scored.append((score, -index, chunk))

        scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
        if not scored:
            return []
        minimum_score = scored[0][0]
        return [
            chunk for score, _, chunk in scored
            if score >= minimum_score
        ][: self.top_k]