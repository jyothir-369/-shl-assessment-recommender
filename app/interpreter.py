"""Conversation interpretation utilities for the SHL Assessment Recommender.

This module converts a full stateless conversation history into structured
signals that the agent can use to:
- clarify vague requests,
- recommend SHL assessments,
- refine prior recommendations,
- compare named assessments,
- refuse off-topic or prompt-injection attempts.

Design goals:
- Deterministic (rule + regex based)
- Stateless (uses only provided message history)
- Robust against vague prompts and hallucination
- SHL assignment aligned
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Any


# ---------------------------------------------------------------------------
# Optional imports from catalog
# ---------------------------------------------------------------------------
try:
    from .catalog import CatalogStore, normalize_text
except Exception:
    # Fallback normalization for standalone safety
    def normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().strip())

    CatalogStore = Any


# ---------------------------------------------------------------------------
# Intent Definitions
# ---------------------------------------------------------------------------
class Intent(str, Enum):
    """Supported conversational intents."""

    CLARIFY = "clarify"
    RECOMMEND = "recommend"
    REFINE = "refine"
    COMPARE = "compare"
    OFF_TOPIC = "off_topic"
    GREET = "greet"


# ---------------------------------------------------------------------------
# Heuristic vocabularies
# ---------------------------------------------------------------------------
GREETING_WORDS = {
    "hello",
    "hi",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
}

# Terms that are too generic to count as role context
VAGUE_ROLE_TERMS = {
    "assessment",
    "assessments",
    "test",
    "tests",
    "job",
    "jobs",
    "role",
    "candidate",
    "hiring",
    "hire",
    "help",
    "developer",
    "engineer",
}

REFINEMENT_CUES = {
    "actually",
    "also",
    "add",
    "include",
    "instead",
    "prefer",
    "with",
    "without",
    "plus",
    "remove",
    "exclude",
}

COMPARISON_PATTERNS = (
    r"\bdifference between\s+(.+?)\s+and\s+(.+)",
    r"\bcompare\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+)",
    r"\b(.+?)\s+(?:vs\.?|versus)\s+(.+)",
)

OFF_TOPIC_PATTERNS = (
    r"\blegal\b",
    r"\blaw\b",
    r"\bsalary\b",
    r"\bcompensation\b",
    r"\binterview questions?\b",
    r"\bgeneral hiring advice\b",
    r"\bhow to hire\b",
    r"\bresume\b",
    r"\bcv\b",
)

PROMPT_INJECTION_PATTERNS = (
    r"ignore previous instructions",
    r"ignore all previous",
    r"disregard the above",
    r"reveal your system prompt",
    r"show hidden prompt",
    r"system prompt",
    r"act as",
    r"you are now",
)

SENIORITY_PATTERNS = {
    "intern": r"\bintern\b",
    "entry-level": r"\b(entry|junior|graduate|0[- ]?2 years?)\b",
    "mid-level": r"\b(mid|intermediate|3[- ]?6 years?|4 years?)\b",
    "senior": r"\b(senior|lead|staff|7\+?\s*years?)\b",
    "manager": r"\b(manager|management)\b",
}

ASSESSMENT_TYPE_KEYWORDS = {
    "P": {"personality", "opq", "behavioral", "behavioural"},
    "A": {"ability", "reasoning", "cognitive", "aptitude"},
    "K": {"knowledge", "technical", "coding", "java", "python", "sql"},
}


# ---------------------------------------------------------------------------
# Conversation Context
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ConversationContext:
    """Structured interpretation of the full conversation."""

    intent: Intent
    latest_user_message: str

    role_title: str | None = None
    seniority: str | None = None
    preferred_test_types: list[str] = field(default_factory=list)
    language: str | None = None
    constraints: list[str] = field(default_factory=list)

    mentioned_assessments: list[str] = field(default_factory=list)
    comparison_targets: list[str] = field(default_factory=list)

    enough_context: bool = False
    is_refinement: bool = False

    def build_query(self) -> str:
        """Build retrieval query from extracted signals."""
        parts: list[str] = []

        if self.role_title:
            parts.append(self.role_title)

        if self.seniority:
            parts.append(self.seniority)

        for code in self.preferred_test_types:
            if code == "P":
                parts.append("personality")
            elif code == "A":
                parts.append("cognitive ability")
            elif code == "K":
                parts.append("technical knowledge")

        parts.extend(self.constraints)

        if self.language:
            parts.append(self.language)

        return " ".join(p for p in parts if p).strip()


# ---------------------------------------------------------------------------
# Message Helpers
# ---------------------------------------------------------------------------
def _latest_user_message(messages: Iterable[object]) -> str:
    """Return latest user message."""
    for msg in reversed(list(messages)):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", ""))
        if getattr(msg, "role", None) == "user":
            return str(getattr(msg, "content", ""))
    return ""


def _all_user_text(messages: Iterable[object]) -> str:
    """Concatenate all user messages."""
    parts: list[str] = []

    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            parts.append(str(msg.get("content", "")))
        elif getattr(msg, "role", None) == "user":
            parts.append(str(getattr(msg, "content", "")))

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Detection Functions
# ---------------------------------------------------------------------------
def _detect_off_topic(text: str) -> bool:
    normalized = normalize_text(text)
    return any(re.search(pattern, normalized) for pattern in OFF_TOPIC_PATTERNS)


def _detect_prompt_injection(text: str) -> bool:
    normalized = normalize_text(text)
    return any(re.search(pattern, normalized) for pattern in PROMPT_INJECTION_PATTERNS)


def _detect_greeting(text: str) -> bool:
    return normalize_text(text) in GREETING_WORDS


def _extract_seniority(text: str) -> str | None:
    normalized = normalize_text(text)

    for label, pattern in SENIORITY_PATTERNS.items():
        if re.search(pattern, normalized):
            return label

    return None


def _extract_test_types(text: str) -> list[str]:
    normalized = normalize_text(text)
    found: list[str] = []

    for code, keywords in ASSESSMENT_TYPE_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            found.append(code)

    return found


def _extract_role_title(text: str) -> str | None:
    """Extract likely role title while rejecting vague requests."""
    normalized = normalize_text(text)

    patterns = [
        r"hiring\s+(?:a|an)?\s*(.+?)(?:\s+who|\s+with|\s+for|\s*$)",
        r"looking for\s+(?:a|an)?\s*(.+?)(?:\s+who|\s+with|\s+for|\s*$)",
        r"need\s+(?:a|an)?\s*(.+?)(?:\s+who|\s+with|\s+for|\s*$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized)

        if match:
            role = match.group(1).strip()

            if (
                not role
                or role in VAGUE_ROLE_TERMS
                or all(token in VAGUE_ROLE_TERMS for token in role.split())
            ):
                return None

            if len(role.split()) <= 8:
                return role

    known_roles = [
        "java developer",
        "software engineer",
        "python developer",
        "sales manager",
        "project manager",
        "business analyst",
        "data scientist",
        "customer service representative",
    ]

    for role in known_roles:
        if role in normalized:
            return role

    return None


def _extract_assessment_names(text: str, catalog: CatalogStore) -> list[str]:
    """Find exact assessment names from catalog."""
    if not catalog:
        return []

    normalized = normalize_text(text)
    matches: list[str] = []

    try:
        for entry in catalog.all():
            if normalize_text(entry.name) in normalized:
                matches.append(entry.name)
    except Exception:
        return []

    return sorted(set(matches))


def _detect_comparison(text: str, catalog: CatalogStore) -> list[str]:
    """Detect comparison requests."""
    normalized = normalize_text(text)

    for pattern in COMPARISON_PATTERNS:
        match = re.search(pattern, normalized)

        if not match:
            continue

        targets: list[str] = []

        for group in match.groups():
            group = group.strip()

            if hasattr(catalog, "get_by_name"):
                item = catalog.get_by_name(group)
                if item:
                    targets.append(item.name)

            targets.extend(_extract_assessment_names(group, catalog))

        deduped = sorted(set(targets))

        if len(deduped) >= 2:
            return deduped[:2]

    mentioned = _extract_assessment_names(text, catalog)

    if len(mentioned) >= 2 and any(
        cue in normalized for cue in ("difference", "compare", "versus", " vs ")
    ):
        return mentioned[:2]

    return []


# ---------------------------------------------------------------------------
# Main Interpretation Entry
# ---------------------------------------------------------------------------
def interpret_conversation(
    messages: Iterable[object],
    catalog: CatalogStore,
) -> ConversationContext:
    """Interpret full stateless conversation history."""

    messages_list = list(messages)

    latest_user = _latest_user_message(messages_list)
    all_user_text = _all_user_text(messages_list)

    if not latest_user:
        return ConversationContext(
            intent=Intent.CLARIFY,
            latest_user_message="",
        )

    # Safety
    if _detect_prompt_injection(latest_user) or _detect_off_topic(latest_user):
        return ConversationContext(
            intent=Intent.OFF_TOPIC,
            latest_user_message=latest_user,
        )

    # Greeting
    if _detect_greeting(latest_user):
        return ConversationContext(
            intent=Intent.GREET,
            latest_user_message=latest_user,
        )

    # Comparison
    comparison_targets = _detect_comparison(latest_user, catalog)

    if len(comparison_targets) >= 2:
        return ConversationContext(
            intent=Intent.COMPARE,
            latest_user_message=latest_user,
            comparison_targets=comparison_targets,
            mentioned_assessments=comparison_targets,
            enough_context=True,
        )

    # Extract structured signals
    role_title = _extract_role_title(all_user_text)
    seniority = _extract_seniority(all_user_text)
    preferred_test_types = _extract_test_types(all_user_text)
    mentioned_assessments = _extract_assessment_names(all_user_text, catalog)

    normalized_latest = normalize_text(latest_user)

    constraints: list[str] = []

    if "stakeholder" in all_user_text.lower():
        constraints.append("stakeholder communication")

    if "client" in all_user_text.lower():
        constraints.append("client interaction")

    if "leadership" in all_user_text.lower():
        constraints.append("leadership")

    # Enough context only if meaningful
    enough_context = bool(
        role_title
        or seniority
        or preferred_test_types
        or mentioned_assessments
    )

    # Explicit vague request detection
    is_vague_request = (
        not role_title
        and not seniority
        and not preferred_test_types
        and not mentioned_assessments
    )

    # Refinement
    is_refinement = False

    if len(messages_list) >= 3:
        if any(cue in normalized_latest for cue in REFINEMENT_CUES):
            is_refinement = True

    # Clarify
    if is_vague_request:
        return ConversationContext(
            intent=Intent.CLARIFY,
            latest_user_message=latest_user,
            role_title=role_title,
            seniority=seniority,
            preferred_test_types=preferred_test_types,
            constraints=constraints,
            enough_context=False,
        )

    # Intent
    intent = Intent.RECOMMEND

    if is_refinement:
        intent = Intent.REFINE

    return ConversationContext(
        intent=intent,
        latest_user_message=latest_user,
        role_title=role_title,
        seniority=seniority,
        preferred_test_types=preferred_test_types,
        constraints=constraints,
        mentioned_assessments=mentioned_assessments,
        enough_context=enough_context,
        is_refinement=is_refinement,
    )