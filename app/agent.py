"""Core orchestration logic for the SHL Assessment Recommender."""

from __future__ import annotations

from typing import Iterable

from .catalog import CatalogEntry, CatalogStore, normalize_text
from .interpreter import Intent, interpret_conversation
from .schemas import ChatResponse, Recommendation


TECHNICAL_KEYWORDS = {
    "developer",
    "engineer",
    "software",
    "technical",
    "coding",
    "programming",
    "backend",
    "frontend",
    "cloud",
    "data",
    "python",
    "java",
    "sql",
    "javascript",
}

ABILITY_KEYWORDS = {
    "problem-solving",
    "problem",
    "solving",
    "cognitive",
    "reasoning",
    "ability",
    "aptitude",
    "analysis",
    "analytical",
    "logic",
    "critical",
}

PERSONALITY_KEYWORDS = {
    "stakeholder",
    "communication",
    "collaboration",
    "collaborates",
    "teamwork",
    "personality",
    "fit",
    "leadership",
    "client",
    "customer",
    "influence",
}

TECH_STACK_KEYWORDS = {
    "python": {"python"},
    "java": {"java"},
    "sql": {"sql"},
    "javascript": {"javascript", "js"},
}


class SHLRecommenderAgent:
    """Stateless conversational agent for recommending SHL assessments."""

    def __init__(self, catalog: CatalogStore | None = None) -> None:
        self.catalog = catalog or CatalogStore()

    def chat(self, messages: Iterable[object]) -> ChatResponse:
        """Process full conversation history and return next response."""
        context = interpret_conversation(messages, self.catalog)

        if context.intent == Intent.OFF_TOPIC:
            return ChatResponse(
                reply=(
                    "I can only help with selecting SHL assessments from the SHL "
                    "product catalog. I cannot provide legal advice, salary guidance, "
                    "general hiring advice, or respond to prompt-injection requests."
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        if context.intent == Intent.GREET:
            return ChatResponse(
                reply=(
                    "Hello! Tell me about the role you are hiring for, and I will "
                    "recommend relevant SHL assessments."
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        if context.intent == Intent.CLARIFY or not context.enough_context:
            return ChatResponse(
                reply=self._build_clarifying_question(context),
                recommendations=[],
                end_of_conversation=False,
            )

        if context.intent == Intent.COMPARE:
            return self._handle_compare(context.comparison_targets)

        if context.intent in {Intent.RECOMMEND, Intent.REFINE}:
            return self._handle_recommendation(context)

        return ChatResponse(
            reply=(
                "Please describe the role you are hiring for and any preferences "
                "such as seniority or desired assessment types."
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    def _build_clarifying_question(self, context) -> str:
        """Ask focused clarification."""
        if not context.role_title:
            return "What role are you hiring for?"
        if not context.seniority:
            return "What is the seniority level for this role?"
        return (
            "Do you need technical, cognitive ability, personality, "
            "or a combination of these assessments?"
        )

    def _detect_role_signals(self, query: str) -> tuple[set[str], str | None]:
        """Detect desired assessment families and explicit technical domain."""
        normalized = normalize_text(query)
        detected: set[str] = set()
        explicit_stack: str | None = None

        if any(keyword in normalized for keyword in TECHNICAL_KEYWORDS):
            detected.add("K")

        if any(keyword in normalized for keyword in ABILITY_KEYWORDS):
            detected.add("A")

        if any(keyword in normalized for keyword in PERSONALITY_KEYWORDS):
            detected.add("P")

        for stack, keywords in TECH_STACK_KEYWORDS.items():
            if any(keyword in normalized for keyword in keywords):
                explicit_stack = stack
                break

        return detected, explicit_stack

    def _is_relevant_technical_match(
        self,
        entry: CatalogEntry,
        explicit_stack: str | None,
    ) -> bool:
        """Ensure technical results align with explicit stack when specified."""
        if entry.test_type != "K":
            return True

        if not explicit_stack:
            return True

        entry_name = normalize_text(entry.name)

        allowed_keywords = TECH_STACK_KEYWORDS.get(explicit_stack, set())

        return any(keyword in entry_name for keyword in allowed_keywords)

    def _collect_candidates(
        self,
        query: str,
        seniority: str | None,
        desired_types: set[str],
        explicit_stack: str | None,
    ) -> list[CatalogEntry]:
        """Collect relevant candidates with domain-aware filtering."""
        collected: list[CatalogEntry] = []
        seen: set[str] = set()

        # Broad retrieval
        broad = self.catalog.retrieve(query, top_k=15, job_level=seniority)

        # Typed reinforcement
        typed_pool: list[CatalogEntry] = []
        for test_type in sorted(desired_types):
            typed_pool.extend(
                self.catalog.retrieve(
                    query,
                    top_k=8,
                    test_type=test_type,
                    job_level=seniority,
                )
            )

        for entry in broad + typed_pool:
            if entry.name in seen:
                continue

            if not self._is_relevant_technical_match(entry, explicit_stack):
                continue

            seen.add(entry.name)
            collected.append(entry)

        return collected

    def _diversify_recommendations(
        self,
        candidates: list[CatalogEntry],
        desired_types: set[str],
        max_results: int = 5,
    ) -> list[CatalogEntry]:
        """Build precise shortlist with strong family coverage and low noise."""
        selected: list[CatalogEntry] = []
        used_names: set[str] = set()

        # Best per family first
        for family in ["K", "A", "P"]:
            if family not in desired_types:
                continue

            for entry in candidates:
                if entry.test_type == family and entry.name not in used_names:
                    selected.append(entry)
                    used_names.add(entry.name)
                    break

        # Strict extras only
        for entry in candidates:
            if len(selected) >= max_results:
                break

            if entry.name in used_names:
                continue

            if entry.test_type not in desired_types:
                continue

            # Prevent family overload
            if sum(1 for e in selected if e.test_type == entry.test_type) >= 2:
                continue

            selected.append(entry)
            used_names.add(entry.name)

        return selected[:max_results]

    def _handle_recommendation(self, context) -> ChatResponse:
        """Retrieve and return grounded, precise shortlist."""
        query = context.build_query()

        if not query:
            return ChatResponse(
                reply="What role are you hiring for?",
                recommendations=[],
                end_of_conversation=False,
            )

        desired_types, explicit_stack = self._detect_role_signals(query)

        if not desired_types:
            candidates = self.catalog.retrieve(
                query,
                top_k=5,
                job_level=context.seniority,
            )
        else:
            candidates = self._collect_candidates(
                query,
                context.seniority,
                desired_types,
                explicit_stack,
            )

        if not candidates:
            return ChatResponse(
                reply=(
                    "I could not find a strong match in the SHL catalog. "
                    "Please provide more details about the role or desired skills."
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        max_results = 5 if len(desired_types) >= 2 else 3

        final_entries = self._diversify_recommendations(
            candidates,
            desired_types,
            max_results=max_results,
        )

        recommendations = [
            Recommendation(
                name=entry.name,
                url=entry.url,
                test_type=entry.test_type,
            )
            for entry in final_entries
            if self.catalog.is_catalog_url(entry.url)
        ]

        if context.is_refinement:
            reply = (
                f"Updated the shortlist based on your new requirements. "
                f"Here are {len(recommendations)} SHL assessments to consider."
            )
        else:
            reply = (
                f"Based on the role requirements, here are "
                f"{len(recommendations)} SHL assessments to consider."
            )

        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=True,
        )

    def _handle_compare(self, assessment_names: list[str]) -> ChatResponse:
        """Compare two named assessments using catalog-only data."""
        entries: list[CatalogEntry] = []

        for name in assessment_names[:2]:
            entry = self.catalog.get_by_name(name)
            if entry:
                entries.append(entry)

        if len(entries) < 2:
            return ChatResponse(
                reply=(
                    "I could not find both assessments in the SHL catalog. "
                    "Please provide the exact assessment names."
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        first, second = entries

        first_desc = first.description or first.category or "an SHL assessment"
        second_desc = second.description or second.category or "an SHL assessment"

        reply = (
            f"{first.name} (type {first.test_type}) focuses on {first_desc}. "
            f"{second.name} (type {second.test_type}) focuses on {second_desc}. "
            f"Choose based on which capabilities are most important for the role."
        )

        return ChatResponse(
            reply=reply,
            recommendations=[],
            end_of_conversation=False,
        )