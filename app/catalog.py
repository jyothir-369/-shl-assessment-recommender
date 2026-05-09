"""Catalog loading and retrieval utilities for the SHL Assessment Recommender.

This module provides a lightweight, dependency-free retrieval layer over the
local ``catalog.json`` file. The catalog is restricted to SHL Individual Test
Solutions, and every recommendation returned by this module is guaranteed to
originate from that file.

Retrieval uses a deterministic hybrid lexical ranking approach:
1. Exact name matching
2. Named assessment / comparison boosts
3. Keyword overlap (weighted)
4. Role-intent weighting (technical / ability / personality)
5. Simple token-frequency scoring over searchable text

The implementation avoids heavyweight dependencies such as vector databases or
embedding models, which keeps startup fast and deployment simple.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


SHL_DOMAIN = "www.shl.com"

# Common English stop words to reduce noise in lexical matching.
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "hiring",
    "i",
    "in",
    "is",
    "it",
    "level",
    "me",
    "need",
    "of",
    "on",
    "or",
    "role",
    "someone",
    "test",
    "tests",
    "the",
    "to",
    "with",
    "years",
}

# Maintainable keyword groups for role-intent scoring.
TECHNICAL_KEYWORDS = {
    "developer",
    "engineer",
    "software",
    "technical",
    "coding",
    "programming",
    "programmer",
    "java",
    "python",
    "sql",
    "developer",
    "backend",
    "frontend",
}

ABILITY_KEYWORDS = {
    "problem",
    "solving",
    "problem-solving",
    "cognitive",
    "reasoning",
    "aptitude",
    "ability",
    "analysis",
    "analytical",
    "logic",
}

PERSONALITY_KEYWORDS = {
    "stakeholder",
    "stakeholders",
    "communication",
    "collaboration",
    "collaborates",
    "teamwork",
    "personality",
    "behavioral",
    "behavioural",
    "motivation",
    "leadership",
    "fit",
}

COMPARISON_KEYWORDS = {
    "compare",
    "comparison",
    "versus",
    "vs",
    "difference",
}


def normalize_text(text: str) -> str:
    """Normalize text for robust lexical matching."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Tokenize normalized text and remove common stop words."""
    normalized = normalize_text(text)
    if not normalized:
        return []

    return [token for token in normalized.split() if token not in STOPWORDS]


def is_valid_catalog_url(url: str) -> bool:
    """Return True only for SHL catalog URLs."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.scheme not in {"http", "https"}:
        return False

    return parsed.netloc.endswith("shl.com")


def extract_test_type(raw: dict[str, Any]) -> str:
    """Extract a test type code from a catalog row."""
    for key in ("test_type", "type", "assessment_type", "code"):
        value = raw.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip().upper()

    searchable = normalize_text(
        " ".join(
            str(raw.get(key, ""))
            for key in ("name", "description", "category", "family")
        )
    )

    if "personality" in searchable or "opq" in searchable:
        return "P"

    if (
        "ability" in searchable
        or "reasoning" in searchable
        or "cognitive" in searchable
    ):
        return "A"

    if (
        "knowledge" in searchable
        or "technical" in searchable
        or "java" in searchable
        or "python" in searchable
        or "coding" in searchable
    ):
        return "K"

    return "U"


def _contains_any(tokens: set[str], keywords: set[str]) -> bool:
    """Return True if token set intersects with keyword group."""
    return bool(tokens & keywords)


@dataclass(slots=True)
class CatalogEntry:
    """Represents a single SHL catalog entry."""

    name: str
    url: str
    test_type: str
    description: str = ""
    category: str = ""
    job_levels: list[str] = field(default_factory=list)
    searchable_text: str = ""
    tokens: set[str] = field(default_factory=set)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CatalogEntry":
        """Create an entry from a JSON row."""
        name = str(raw.get("name", "")).strip()
        url = str(raw.get("url", "")).strip()

        if not name:
            raise ValueError("Catalog entry is missing 'name'.")

        if not is_valid_catalog_url(url):
            raise ValueError(f"Invalid catalog URL for entry '{name}'.")

        description = str(raw.get("description", "")).strip()
        category = str(raw.get("category", "")).strip()

        raw_levels = raw.get("job_levels", [])

        if isinstance(raw_levels, str):
            job_levels = [
                level.strip()
                for level in raw_levels.split(",")
                if level.strip()
            ]

        elif isinstance(raw_levels, list):
            job_levels = [
                str(level).strip()
                for level in raw_levels
                if str(level).strip()
            ]

        else:
            job_levels = []

        searchable_parts = [
            name,
            description,
            category,
            " ".join(job_levels),
            str(raw.get("skills", "")),
            str(raw.get("family", "")),
        ]

        searchable_text = " ".join(
            part for part in searchable_parts if part
        ).strip()

        return cls(
            name=name,
            url=url,
            test_type=extract_test_type(raw),
            description=description,
            category=category,
            job_levels=job_levels,
            searchable_text=searchable_text,
            tokens=set(tokenize(searchable_text)),
        )

    def to_recommendation(self) -> dict[str, str]:
        """Return the exact schema expected by the API response."""
        return {
            "name": self.name,
            "url": self.url,
            "test_type": self.test_type,
        }


class CatalogStore:
    """Loads and retrieves entries from the local SHL catalog."""

    def __init__(self, catalog_path: str | Path = "catalog.json") -> None:
        self.catalog_path = Path(catalog_path)
        self._entries: list[CatalogEntry] = []
        self._by_name: dict[str, CatalogEntry] = {}
        self.load()

    def load(self) -> None:
        """Load catalog entries from disk."""
        if not self.catalog_path.exists():
            raise FileNotFoundError(
                f"Catalog file not found: {self.catalog_path}"
            )

        with self.catalog_path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raise ValueError("catalog.json must contain a JSON array.")

        entries: list[CatalogEntry] = []

        for item in raw_data:
            if not isinstance(item, dict):
                continue

            try:
                entries.append(CatalogEntry.from_dict(item))
            except ValueError:
                continue

        if not entries:
            raise ValueError("No valid catalog entries were loaded.")

        self._entries = entries
        self._by_name = {
            normalize_text(entry.name): entry for entry in entries
        }

    def all(self) -> list[CatalogEntry]:
        """Return all catalog entries."""
        return list(self._entries)

    def get_by_name(self, name: str) -> CatalogEntry | None:
        """Retrieve an entry by exact normalized name."""
        return self._by_name.get(normalize_text(name))

    def search_by_keywords(
        self,
        keywords: str | Iterable[str],
    ) -> list[CatalogEntry]:
        """Retrieve entries containing one or more keywords."""
        if isinstance(keywords, str):
            query_tokens = set(tokenize(keywords))

        else:
            query_tokens = {
                token
                for keyword in keywords
                for token in tokenize(str(keyword))
            }

        if not query_tokens:
            return []

        matches: list[CatalogEntry] = []

        for entry in self._entries:
            if query_tokens & entry.tokens:
                matches.append(entry)

        return matches

    def filter_entries(
        self,
        entries: Iterable[CatalogEntry],
        *,
        test_type: str | None = None,
        job_level: str | None = None,
    ) -> list[CatalogEntry]:
        """Filter entries by optional metadata."""
        filtered = list(entries)

        if test_type:
            code = test_type.strip().upper()
            filtered = [
                entry for entry in filtered if entry.test_type == code
            ]

        if job_level:
            level = normalize_text(job_level)

            filtered = [
                entry
                for entry in filtered
                if any(
                    level in normalize_text(job_level_value)
                    for job_level_value in entry.job_levels
                )
            ]

        return filtered

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        test_type: str | None = None,
        job_level: str | None = None,
    ) -> list[CatalogEntry]:
        """Retrieve the most relevant catalog entries.

        Ranking combines:
        - Exact title match
        - Named assessment boost
        - Title token overlap
        - Searchable text overlap
        - Role-intent weighting
        - Phrase containment
        """
        query_tokens = set(tokenize(query))

        if not query_tokens:
            return []

        normalized_query = normalize_text(query)

        technical_intent = _contains_any(query_tokens, TECHNICAL_KEYWORDS)
        ability_intent = _contains_any(query_tokens, ABILITY_KEYWORDS)
        personality_intent = _contains_any(query_tokens, PERSONALITY_KEYWORDS)
        comparison_intent = _contains_any(query_tokens, COMPARISON_KEYWORDS)

        scored: list[tuple[float, CatalogEntry]] = []

        for entry in self._entries:
            score = 0.0

            normalized_name = normalize_text(entry.name)
            normalized_text = normalize_text(entry.searchable_text)

            name_tokens = set(tokenize(entry.name))

            # 1. Strongest possible exact assessment match.
            if normalized_query == normalized_name:
                score += 100.0

            # 2. Direct named assessment mention boost.
            if normalized_name in normalized_query:
                score += 60.0

            # 3. Comparison queries should prioritize exact named matches.
            if comparison_intent and (query_tokens & name_tokens):
                score += 25.0

            # 4. Title overlap.
            title_overlap = len(query_tokens & name_tokens)
            score += title_overlap * 12.0

            # 5. Searchable text overlap.
            body_overlap = len(query_tokens & entry.tokens)
            score += body_overlap * 4.0

            # 6. Phrase containment.
            if normalized_query and normalized_query in normalized_text:
                score += 15.0

            # 7. Technical weighting.
            if technical_intent:
                if entry.test_type == "K":
                    score += 20.0

                technical_overlap = len(
                    entry.tokens & TECHNICAL_KEYWORDS
                )

                score += technical_overlap * 3.0

            # 8. Ability weighting.
            if ability_intent:
                if entry.test_type == "A":
                    score += 18.0

                ability_overlap = len(
                    entry.tokens & ABILITY_KEYWORDS
                )

                score += ability_overlap * 3.0

            # 9. Personality weighting.
            if personality_intent:
                if entry.test_type == "P":
                    score += 16.0

                personality_overlap = len(
                    entry.tokens & PERSONALITY_KEYWORDS
                )

                score += personality_overlap * 3.0

            # 10. Prevent technical roles from collapsing into personality-only.
            if technical_intent and entry.test_type == "P":
                score -= 4.0

            if score > 0:
                scored.append((score, entry))

        # Deterministic ordering.
        scored.sort(
            key=lambda item: (-item[0], item[1].name)
        )

        ranked_entries = [entry for _, entry in scored]

        ranked_entries = self.filter_entries(
            ranked_entries,
            test_type=test_type,
            job_level=job_level,
        )

        return ranked_entries[: max(1, min(top_k, 10))]

    def catalog_urls(self) -> set[str]:
        """Return the set of all allowed catalog URLs."""
        return {entry.url for entry in self._entries}

    def is_catalog_url(self, url: str) -> bool:
        """Return True only if the URL exists in the loaded catalog."""
        return url in self.catalog_urls()