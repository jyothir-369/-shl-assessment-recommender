"""Core orchestration logic for the SHL Assessment Recommender."""

from __future__ import annotations

from typing import Iterable

from .catalog import CatalogEntry, CatalogStore, normalize_text
from .interpreter import Intent, interpret_conversation
from .schemas import ChatResponse, Recommendation


TECHNICAL_SIGNALS = {
    "developer",
    "engineer",
    "technical",
    "coding",
    "programming",
    "software",
    "java",
    "python",
    "sql",
    "backend",
    "frontend",
}

ABILITY_SIGNALS = {
    "problem",
    "solving",
    "problem-solving",
    "cognitive",
    "reasoning",
    "ability",
    "analytical",
    "logic",
}

PERSONALITY_SIGNALS = {
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


class SHLRecommenderAgent:
    """Stateless conversational agent for recommending SHL assessments."""

    def __init__(self, catalog: CatalogStore | None = None) -> None:
        self.catalog = catalog or CatalogStore()

    def chat(self, messages: Iterable[object]) -> ChatResponse:
        """Process full stateless conversation history."""
        context = interpret_conversation(messages, self.catalog)

        # Refusal / off-topic
        if context.intent == Intent.OFF_TOPIC:
            return ChatResponse(
                reply=(
                    "I can only help select SHL assessments from the SHL "
                    "catalog. I cannot assist with legal, salary, general "
                    "hiring advice, or prompt-injection requests."
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        # Greeting
        if context.intent == Intent.GREET:
            return ChatResponse(
                reply=(
                    "Hello! Describe the role, seniority, and required skills, "
                    "and I’ll recommend relevant SHL assessments."
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        # Clarification
        if context.intent == Intent.CLARIFY or not context.enough_context:
            return ChatResponse(
                reply=self._build_clarifying_question(context),
                recommendations=[],
                end_of_conversation=False,
            )

        # Comparison
        if context.intent == Intent.COMPARE:
            return self._handle_compare(context.comparison_targets)

        # Recommendation / refinement
        if context.intent in {Intent.RECOMMEND, Intent.REFINE}:
            return self._handle_recommendation(context)

        # Safe fallback
        return ChatResponse(
            reply=(
                "Please describe the role and whether you need technical, "
                "cognitive, or personality assessments."
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    def _build_clarifying_question(self, context) -> str:
        """Ask one focused clarifying question."""
        if not context.role_title:
            return "What role are you hiring for?"
        if not context.seniority:
            return "What seniority level is the role?"
        return (
            "Do you need technical/coding, cognitive/problem-solving, "
            "personality, or a combination?"
        )

    def _detect_role_signals(
        self,
        context,
    ) -> tuple[bool, bool, bool]:
        """Detect whether the role implies K / A / P needs."""
        query_text = normalize_text(
            f"{context.latest_user_message} {context.build_query()}"
        )

        wants_technical = any(
            signal in query_text for signal in TECHNICAL_SIGNALS
        )

        wants_ability = any(
            signal in query_text for signal in ABILITY_SIGNALS
        )

        wants_personality = any(
            signal in query_text for signal in PERSONALITY_SIGNALS
        )

        # Explicit preferred test types reinforce signals.
        if "K" in context.preferred_test_types:
            wants_technical = True

        if "A" in context.preferred_test_types:
            wants_ability = True

        if "P" in context.preferred_test_types:
            wants_personality = True

        return wants_technical, wants_ability, wants_personality

    def _diversify_recommendations(
        self,
        candidates: list[CatalogEntry],
        *,
        wants_technical: bool,
        wants_ability: bool,
        wants_personality: bool,
        max_results: int = 10,
    ) -> list[CatalogEntry]:
        """Ensure balanced shortlist without sacrificing ranking."""
        if not candidates:
            return []

        selected: list[CatalogEntry] = []
        used_names: set[str] = set()

        def add_best_of_type(test_type: str) -> None:
            for entry in candidates:
                if (
                    entry.test_type == test_type
                    and entry.name not in used_names
                ):
                    selected.append(entry)
                    used_names.add(entry.name)
                    return

        # Seed shortlist with strongest category coverage first.
        if wants_technical:
            add_best_of_type("K")

        if wants_ability:
            add_best_of_type("A")

        if wants_personality:
            add_best_of_type("P")

        # Fill remaining slots by original ranking order.
        for entry in candidates:
            if entry.name in used_names:
                continue

            selected.append(entry)
            used_names.add(entry.name)

            if len(selected) >= max_results:
                break

        return selected[:max_results]

    def _validated_recommendations(
        self,
        entries: list[CatalogEntry],
    ) -> list[Recommendation]:
        """Convert catalog entries into validated API recommendations."""
        recommendations: list[Recommendation] = []

        for entry in entries:
            if not self.catalog.is_catalog_url(entry.url):
                continue

            recommendations.append(
                Recommendation(
                    name=entry.name,
                    url=entry.url,
                    test_type=entry.test_type,
                )
            )

        return recommendations[:10]

    def _handle_recommendation(self, context) -> ChatResponse:
        """Retrieve and assemble a balanced, grounded shortlist."""
        query = context.build_query()

        if not query:
            return ChatResponse(
                reply="What role are you hiring for?",
                recommendations=[],
                end_of_conversation=False,
            )

        # IMPORTANT:
        # Broad retrieval first — never over-filter to one type too early.
        candidates = self.catalog.retrieve(
            query,
            top_k=10,
            job_level=context.seniority,
        )

        # Fallback broader search.
        if not candidates:
            candidates = self.catalog.retrieve(query, top_k=10)

        if not candidates:
            return ChatResponse(
                reply=(
                    "I could not find strong SHL catalog matches. "
                    "Please provide more role details."
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        wants_technical, wants_ability, wants_personality = (
            self._detect_role_signals(context)
        )

        diversified = self._diversify_recommendations(
            candidates,
            wants_technical=wants_technical,
            wants_ability=wants_ability,
            wants_personality=wants_personality,
            max_results=10,
        )

        recommendations = self._validated_recommendations(diversified)

        if not recommendations:
            return ChatResponse(
                reply=(
                    "I could not produce valid SHL recommendations. "
                    "Please refine your request."
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        if context.intent == Intent.REFINE or context.is_refinement:
            reply = (
                f"Updated your shortlist based on refined requirements. "
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

    def _handle_compare(
        self,
        assessment_names: list[str],
    ) -> ChatResponse:
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
                    "Please provide exact assessment names."
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        first, second = entries

        first_desc = first.description or first.category or "its target domain"
        second_desc = second.description or second.category or "its target domain"

        reply = (
            f"{first.name} (type {first.test_type}) focuses on {first_desc}. "
            f"{second.name} (type {second.test_type}) focuses on {second_desc}. "
            f"Choose based on which capabilities are most relevant for your role."
        )

        return ChatResponse(
            reply=reply,
            recommendations=[],
            end_of_conversation=False,
        )