"""Core orchestration logic for the SHL Assessment Recommender."""

from __future__ import annotations

import re
from collections import defaultdict
from enum import Enum, auto
from typing import Iterable

from .catalog import CatalogEntry, CatalogStore, normalize_text
from .interpreter import Intent, interpret_conversation
from .schemas import ChatResponse, Recommendation


# ---------------------------------------------------------------------------
# Family mode enum
# ---------------------------------------------------------------------------

class FamilyMode(Enum):
    K_ONLY  = auto()
    A_ONLY  = auto()
    P_ONLY  = auto()
    MIXED   = auto()   # 2-family or 3-family explicit
    UNKNOWN = auto()   # no explicit signal — fall back to K


# ---------------------------------------------------------------------------
# Keyword banks
# ---------------------------------------------------------------------------

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
    "api",
    "apis",
    "microservices",
    "data",
    "python",
    "java",
    "sql",
    "javascript",
}

ABILITY_KEYWORDS = {
    "problem-solving",
    "cognitive",
    "reasoning",
    "ability",
    "aptitude",
    "analytical",
    "logic",
    "critical",
    "graduate",
}

PERSONALITY_KEYWORDS = {
    "stakeholder",
    "communication",
    "collaboration",
    "teamwork",
    "personality",
    "leadership",
    "customer-facing",
    "interpersonal",
    "people manager",
    "people managers",
    "behavioural",
    "behavioral",
    "motivat",
}

# EXPLICIT signals — strong domain terms that must appear for a family to be
# considered explicitly requested.
EXPLICIT_P_SIGNALS = {
    "personality",
    "collaboration",
    "teamwork",
    "interpersonal",
    "people manager",
    "people managers",
    "leadership",
    "behavioural",
    "behavioral",
    "communication",
    "stakeholder",
    "motivat",
}

EXPLICIT_A_SIGNALS = {
    "cognitive",
    "reasoning",
    "analytical",
    "aptitude",
    "graduate",
    "ability",
    "logic",
    "problem-solving",
}

EXPLICIT_K_SIGNALS = {
    "developer",
    "engineer",
    "software",
    "coding",
    "programming",
    "backend",
    "frontend",
    "python",
    "java",
    "sql",
    "javascript",
    "api",
    "apis",
    "technical",
    "microservices",
}

TECH_STACK_KEYWORDS: dict[str, set[str]] = {
    "python":     {"python"},
    "java":       {"java"},
    "sql":        {"sql"},
    "javascript": {"javascript", "js"},
}

STACK_EXCLUSIONS: dict[str, set[str]] = {
    "java":       {"javascript", "python"},
    "python":     {"java", "javascript"},
    "sql":        {"java", "javascript", "python"},
    "javascript": {"java", "python"},
}

PREFERRED_P_TYPES = {
    "opq",
    "personality",
    "motivation",
    "occupational",
    "behavior",
    "behaviour",
}

# ---------------------------------------------------------------------------
# SJT / situational-judgement suppression
# ---------------------------------------------------------------------------

SJT_NOISE_TERMS = {
    "situational judgement",
    "situational judgment",
    "sjt",
}

# Only allow SJT when the user explicitly asks for one of these.
SJT_REQUESTED_SIGNALS = {
    "situational judgement",
    "situational judgment",
    "sjt",
    "managerial judgment",
    "managerial judgement",
    "leadership judgment",
    "leadership judgement",
}

# ---------------------------------------------------------------------------
# Refusal terms
# ---------------------------------------------------------------------------

REFUSAL_TERMS = {
    "ignore your instructions",
    "ignore previous instructions",
    "salary",
    "compensation",
    "aws certification",
    "aws certifications",
    "legal advice",
}

# ---------------------------------------------------------------------------
# Compare intent detection
# ---------------------------------------------------------------------------

COMPARE_LEAD_PHRASES = {
    "compare",
    "difference between",
    "differences between",
    "contrast",
}

COMPARE_SEPARATOR_TERMS = {"vs", "versus"}

# ---------------------------------------------------------------------------
# Alias map  (normalised key -> exact catalog name)
# ---------------------------------------------------------------------------

COMPARE_ALIASES: dict[str, str] = {
    "opq": "Occupational Personality Questionnaire (OPQ32r)",
    "opq32": "Occupational Personality Questionnaire (OPQ32r)",
    "opq32r": "Occupational Personality Questionnaire (OPQ32r)",
    "occupational personality questionnaire": "Occupational Personality Questionnaire (OPQ32r)",
    "occupational personality questionnaire opq32r": "Occupational Personality Questionnaire (OPQ32r)",
    "gsa": "General Ability Screen",
    "general ability screen": "General Ability Screen",
    "general ability": "General Ability Screen",
    "mq": "Motivation Questionnaire (MQ)",
    "motivation questionnaire": "Motivation Questionnaire (MQ)",
    "motivation questionnaire mq": "Motivation Questionnaire (MQ)",
    "verify interactive g+": "Verify Interactive G+",
    "verify interactive g": "Verify Interactive G+",
    "verify g+": "Verify Interactive G+",
    "verify g": "Verify Interactive G+",
    "verify interactive": "Verify Interactive G+",
    "numerical reasoning": "Verify - Numerical Ability",
    "verbal reasoning": "Verify - Verbal Ability",
    "deductive reasoning": "Verify - Deductive Reasoning",
    "ccsq": "Contact Center Simulation",
    "contact center simulation": "Contact Center Simulation",
}

# ---------------------------------------------------------------------------
# Noise / filler suppression
# ---------------------------------------------------------------------------

NOISY_SIMULATION_TERMS = {
    "simulation",
    "contact center",
    "contact centre",
    "customer service",
    "customer support",
    "sales achievement",
    "sales predictor",
    "sapa",
    "call center",
    "call centre",
}

# ---------------------------------------------------------------------------
# [CHANGED] Expanded simulation requested signals — more robust detection
# ---------------------------------------------------------------------------
# Previously this set was too narrow. Expanded to catch common phrasings like
# "call center hiring", "customer service simulation", "contact centre", etc.
NOISY_SIMULATION_REQUESTED_SIGNALS = {
    "contact center",
    "contact centre",
    "customer service",
    "customer support",
    "sales",
    "simulation",
    "call center",
    "call centre",
    # Additional explicit triggers added:
    "customer-facing",
    "service simulation",
    "center simulation",
    "centre simulation",
    "ccsq",
}

# [NEW] Terms used to identify simulation-type catalog entries for boosting
SIMULATION_BOOST_TERMS = {
    "simulation",
    "contact center",
    "contact centre",
    "customer service",
    "customer support",
    "call center",
    "call centre",
    "sales",
    "service profile",
}

# [NEW] Dedicated retrieval queries used when simulation is explicitly requested
SIMULATION_RETRIEVAL_QUERIES = [
    "contact center simulation",
    "customer service simulation",
    "call center assessment",
    "sales simulation",
    "customer support profile",
    "contact centre",
]

# ---------------------------------------------------------------------------
# Refinement clause markers
# ---------------------------------------------------------------------------

REFINEMENT_MARKERS: list[str] = [
    "actually",
    "instead",
    "rather",
    "now it is",
    "now its",
    "now for",
    "focus on",
    "this is for",
    "changed to",
    "switch to",
    "switching to",
    "update to",
    "updating to",
    "make it",
    "make this",
]

# ---------------------------------------------------------------------------
# Vague-request detection
# ---------------------------------------------------------------------------

VAGUE_EXACT_STRIPPED: set[str] = {
    "im hiring",
    "i m hiring",
    "hiring",
    "we are hiring",
    "were hiring",
    "looking to hire",
    "i need an assessment",
    "we need an assessment",
    "need an assessment",
    "i need assessments",
    "we need assessments",
    "need assessments",
    "i want an assessment",
    "we want an assessment",
    "i am hiring",
}

VAGUE_PATTERNS_STRIPPED: list[re.Pattern] = [
    re.compile(r"^(i\s*m|i\s+am|we\s+are|were?)?\s*hir(ing|e)\s*$"),
    re.compile(r"^looking\s+to\s+hire\s*$"),
    re.compile(r"^(i|we)?\s*(need|want)\s*(an?\s*)?assessments?\s*$"),
]

VAGUE_CLARIFICATION_REPLY = (
    "What role are you hiring for, and which technical, problem-solving, "
    "cognitive, or interpersonal skills matter most?"
)

# ---------------------------------------------------------------------------
# Off-topic signal detection
# ---------------------------------------------------------------------------

OFF_TOPIC_SIGNALS = {
    "streamlit",
    "test harness",
    "manual test",
    "dockerfile",
    "docker",
    "kubectl",
    "nginx",
    "redis",
    "celery",
    "localhost",
    "curl",
    "bash",
    "ci/cd",
    "webpack",
    "pytest",
    "unittest",
    "linting",
    "lint",
    "git commit",
    "pull request",
}

OFF_TOPIC_LENGTH_THRESHOLD = 20


# ---------------------------------------------------------------------------
# Module-level normalisation helpers
# ---------------------------------------------------------------------------

def _normalize_for_vague(text: str) -> str:
    lowered = text.lower()
    stripped = re.sub(r"[^a-z0-9\s]", "", lowered)
    return " ".join(stripped.split())


def _split_into_clauses(text: str) -> list[str]:
    raw_sentences = re.split(r"[.!?]+", text)

    clauses: list[str] = []
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        lower = sentence.lower()

        split_positions: list[int] = [0]
        for marker in REFINEMENT_MARKERS:
            idx = lower.find(marker)
            if idx != -1:
                tail_start = idx + len(marker)
                if tail_start < len(sentence):
                    split_positions.append(tail_start)

        split_positions = sorted(set(split_positions))

        if len(split_positions) == 1:
            clauses.append(sentence)
        else:
            for i, pos in enumerate(split_positions):
                end = split_positions[i + 1] if i + 1 < len(split_positions) else len(sentence)
                chunk = sentence[pos:end].strip()
                if chunk:
                    clauses.append(chunk)

    return clauses


def _find_rightmost_stack_in_text(text: str) -> str | None:
    normalized = normalize_text(text)
    last_found: str | None = None
    last_pos: int = -1

    for stack, keywords in TECH_STACK_KEYWORDS.items():
        for kw in keywords:
            pos = normalized.rfind(kw)
            if pos != -1 and pos > last_pos:
                last_pos = pos
                last_found = stack

    return last_found


# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------

class SHLRecommenderAgent:
    """Stateless conversational SHL assessment recommender."""

    def __init__(self, catalog: CatalogStore | None = None) -> None:
        self.catalog = catalog or CatalogStore()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def chat(self, messages: Iterable[object]) -> ChatResponse:
        messages = list(messages)

        latest_message = ""
        if messages:
            latest = messages[-1]
            if isinstance(latest, dict):
                latest_message = latest.get("content", "")
            else:
                latest_message = getattr(latest, "content", "")

        latest_normalized = normalize_text(latest_message)

        # PRIORITY 1 — Refusal
        if self._is_refusal_request(latest_normalized):
            return ChatResponse(
                reply=(
                    "I can only help with recommending SHL assessments "
                    "from the SHL product catalog."
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        # PRIORITY 2 — Off-topic content
        if self._is_off_topic(latest_normalized):
            return ChatResponse(
                reply=(
                    "I can only assist with SHL assessment recommendations. "
                    "Please describe the role you are hiring for."
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        # PRIORITY 3 — Compare mode
        if self._is_compare_request(latest_normalized):
            targets = self._extract_compare_targets(latest_message)
            return self._handle_compare(targets)

        # PRIORITY 4 — Vague latest-turn dominance
        if self._is_vague_request(latest_message):
            return ChatResponse(
                reply=VAGUE_CLARIFICATION_REPLY,
                recommendations=[],
                end_of_conversation=False,
            )

        # Full conversation interpretation
        context = interpret_conversation(messages, self.catalog)

        if context.intent == Intent.GREET:
            return ChatResponse(
                reply=(
                    "Hello! Tell me about the role, required skills, "
                    "and whether you want technical, cognitive, "
                    "personality, or balanced SHL assessments."
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        if context.intent in {Intent.RECOMMEND, Intent.REFINE}:
            return self._handle_recommendation(
                context=context,
                latest_message=latest_message,
            )

        return ChatResponse(
            reply=VAGUE_CLARIFICATION_REPLY,
            recommendations=[],
            end_of_conversation=False,
        )

    # ------------------------------------------------------------------
    # Intent classifiers
    # ------------------------------------------------------------------

    def _is_refusal_request(self, text: str) -> bool:
        return any(term in text for term in REFUSAL_TERMS)

    def _is_off_topic(self, text: str) -> bool:
        if any(signal in text for signal in OFF_TOPIC_SIGNALS):
            return True

        word_count = len(text.split())
        if word_count > OFF_TOPIC_LENGTH_THRESHOLD:
            has_hiring_signal = any(
                kw in text
                for kw in (
                    TECHNICAL_KEYWORDS
                    | ABILITY_KEYWORDS
                    | PERSONALITY_KEYWORDS
                    | {"hire", "hiring", "assess", "assessment", "role", "candidate"}
                )
            )
            if not has_hiring_signal:
                return True

        return False

    def _is_compare_request(self, text: str) -> bool:
        for phrase in COMPARE_LEAD_PHRASES:
            if phrase in text:
                return True

        for sep in COMPARE_SEPARATOR_TERMS:
            pattern = re.compile(
                r"\b[\w\+]+(?:\s[\w\+]+){0,4}\s+" + re.escape(sep) + r"\s+[\w\+]"
            )
            if pattern.search(text):
                return True

        return False

    def _is_vague_request(self, raw_text: str) -> bool:
        vague_norm = _normalize_for_vague(raw_text)

        if vague_norm in VAGUE_EXACT_STRIPPED:
            return True

        for pattern in VAGUE_PATTERNS_STRIPPED:
            if pattern.match(vague_norm):
                return True

        return False

    # ------------------------------------------------------------------
    # [CHANGED] Simulation detection — now more robust
    # ------------------------------------------------------------------
    # Previously only checked NOISY_SIMULATION_REQUESTED_SIGNALS as a flat
    # set lookup. Now also checks multi-word compound phrases explicitly so
    # "customer service call center simulation" reliably returns True.

    def _user_requested_simulations(self, query: str) -> bool:
        """
        Return True when the user has explicitly requested simulation,
        contact-center, or customer-service type assessments.

        Handles compound phrases like:
          "customer service call center simulation"
          "contact center simulation hiring"
          "call center assessment"
          "sales simulation"
        """
        normalized = normalize_text(query)

        # Check the expanded signal set (single-term hits)
        if any(sig in normalized for sig in NOISY_SIMULATION_REQUESTED_SIGNALS):
            return True

        # [NEW] Explicit compound-phrase patterns for robustness
        SIMULATION_COMPOUND_PATTERNS = [
            r"customer\s+service",
            r"call\s+cent(er|re)",
            r"contact\s+cent(er|re)",
            r"service\s+simulation",
            r"center\s+simulation",
            r"centre\s+simulation",
            r"sales\s+simulation",
            r"customer\s+support",
        ]
        for pattern in SIMULATION_COMPOUND_PATTERNS:
            if re.search(pattern, normalized):
                return True

        return False

    # ------------------------------------------------------------------
    # Compare helpers
    # ------------------------------------------------------------------

    def _extract_compare_targets(self, text: str) -> list[str]:
        normalized = normalize_text(text)

        for phrase in ("difference between", "differences between", "compare", "contrast"):
            normalized = normalized.replace(phrase, "")

        for sep in ("versus", " vs ", " and ", ","):
            normalized = normalized.replace(sep, ",")

        normalized = re.sub(r"[^\w\s,\+]", " ", normalized)

        parts = [p.strip() for p in normalized.split(",") if p.strip()]
        return parts[:2]

    def _resolve_assessment_name(self, raw_name: str) -> CatalogEntry | None:
        normalized = normalize_text(raw_name)

        canonical = COMPARE_ALIASES.get(normalized)
        if canonical:
            entry = self.catalog.get_by_name(canonical)
            if entry:
                return entry

        entry = self.catalog.get_by_name(raw_name)
        if entry:
            return entry

        for alias_key, canonical_name in COMPARE_ALIASES.items():
            alias_norm = normalize_text(alias_key)
            if alias_norm in normalized or normalized in alias_norm:
                entry = self.catalog.get_by_name(canonical_name)
                if entry:
                    return entry

        query_tokens = set(normalized.split())
        candidates = self.catalog.retrieve(raw_name, top_k=15)
        best: CatalogEntry | None = None
        best_overlap = 0

        for candidate in candidates:
            entry_norm = normalize_text(candidate.name)
            entry_tokens = set(entry_norm.split())
            overlap = len(query_tokens & entry_tokens)
            if normalized in entry_norm or entry_norm in normalized:
                overlap += 5
            if overlap > best_overlap:
                best_overlap = overlap
                best = candidate

        if best and best_overlap > 0:
            return best

        return None

    # ------------------------------------------------------------------
    # Clarification
    # ------------------------------------------------------------------

    def _build_clarifying_question(self, context) -> str:
        if not context.role_title:
            return VAGUE_CLARIFICATION_REPLY

        if not context.seniority:
            return (
                f"What seniority level is the {context.role_title} role, "
                "and should assessments focus on technical skills, cognitive ability, "
                "personality traits, or a balanced mix?"
            )

        return (
            "Should the recommendations emphasise technical skills, "
            "cognitive ability, personality traits, or a balanced mix?"
        )

    # ------------------------------------------------------------------
    # CLAUSE-AWARE stack extraction
    # ------------------------------------------------------------------

    def _extract_latest_explicit_stacks(self, latest_message: str) -> set[str]:
        clauses = _split_into_clauses(latest_message)

        if not clauses:
            return set()

        for clause in reversed(clauses):
            active = _find_rightmost_stack_in_text(clause)
            if active:
                return {active}

        return set()

    def _rewrite_query_for_latest_stack(
        self,
        query: str,
        explicit_stacks: set[str],
    ) -> str:
        if not explicit_stacks:
            return query

        normalized = normalize_text(query)
        active_stack = next(iter(explicit_stacks))

        for stack, keywords in TECH_STACK_KEYWORDS.items():
            if stack == active_stack:
                continue
            for kw in keywords:
                normalized = re.sub(r"\b" + re.escape(kw) + r"\b", " ", normalized)

        return " ".join(normalized.split())

    def _detect_role_signals(
        self,
        query: str,
    ) -> tuple[dict[str, int], set[str]]:
        normalized = normalize_text(query)
        family_weights: dict[str, int] = defaultdict(int)

        technical_hits   = sum(1 for kw in TECHNICAL_KEYWORDS   if kw in normalized)
        ability_hits     = sum(1 for kw in ABILITY_KEYWORDS      if kw in normalized)
        personality_hits = sum(1 for kw in PERSONALITY_KEYWORDS  if kw in normalized)

        if technical_hits:
            family_weights["K"] += technical_hits * 3
        if ability_hits:
            family_weights["A"] += ability_hits * 5
        if personality_hits:
            family_weights["P"] += personality_hits * 7

        explicit_stacks: set[str] = set()
        for stack, keywords in TECH_STACK_KEYWORDS.items():
            if any(kw in normalized for kw in keywords):
                explicit_stacks.add(stack)

        return dict(family_weights), explicit_stacks

    def _explicit_families(self, query: str) -> set[str]:
        """Return families that are *explicitly* signaled in the query."""
        normalized = normalize_text(query)
        families: set[str] = set()

        if any(sig in normalized for sig in EXPLICIT_K_SIGNALS):
            families.add("K")
        if any(sig in normalized for sig in EXPLICIT_A_SIGNALS):
            families.add("A")
        if any(sig in normalized for sig in EXPLICIT_P_SIGNALS):
            families.add("P")

        return families

    def _derive_family_mode(
        self,
        latest_explicit: set[str],
        latest_weights: dict[str, int],
    ) -> FamilyMode:
        n = len(latest_explicit)

        if n >= 2:
            return FamilyMode.MIXED

        if n == 1:
            fam = next(iter(latest_explicit))
            if fam == "K":
                return FamilyMode.K_ONLY
            if fam == "A":
                return FamilyMode.A_ONLY
            if fam == "P":
                return FamilyMode.P_ONLY

        if latest_weights:
            dominant = max(latest_weights, key=latest_weights.__getitem__)
            if dominant == "K":
                return FamilyMode.K_ONLY
            if dominant == "A":
                return FamilyMode.A_ONLY
            if dominant == "P":
                return FamilyMode.P_ONLY

        return FamilyMode.UNKNOWN

    # ------------------------------------------------------------------
    # Noise detection
    # ------------------------------------------------------------------

    def _user_requested_sjt(self, query: str) -> bool:
        """Return True only when the user explicitly asks for SJT content."""
        normalized = normalize_text(query)
        return any(sig in normalized for sig in SJT_REQUESTED_SIGNALS)

    def _is_sjt_entry(self, entry: CatalogEntry) -> bool:
        """Return True if this catalog entry belongs to the SJT family."""
        text = normalize_text(
            f"{entry.name} {entry.description or ''} {entry.category or ''}"
        )
        return any(term in text for term in SJT_NOISE_TERMS)

    # ------------------------------------------------------------------
    # [NEW] Simulation entry identification
    # ------------------------------------------------------------------
    # Determines whether a catalog entry is a simulation / contact-center
    # type product. Used for targeted boosting when simulation_requested=True.

    def _is_simulation_entry(self, entry: CatalogEntry) -> bool:
        """
        Return True if this catalog entry is a simulation / contact-center
        / customer-service type assessment.
        """
        text = normalize_text(
            f"{entry.name} {entry.description or ''} {entry.category or ''}"
        )
        return any(term in text for term in SIMULATION_BOOST_TERMS)

    # ------------------------------------------------------------------
    # Technical relevance filter
    # ------------------------------------------------------------------

    def _is_relevant_technical_match(
        self,
        entry: CatalogEntry,
        explicit_stacks: set[str],
    ) -> bool:
        if entry.test_type != "K":
            return True
        if not explicit_stacks:
            return True

        text = normalize_text(
            f"{entry.name} {entry.description or ''} {entry.category or ''}"
        )
        active_stack = next(iter(explicit_stacks))
        allowed_keywords = TECH_STACK_KEYWORDS.get(active_stack, set())

        if not any(kw in text for kw in allowed_keywords):
            return False

        blocked = STACK_EXCLUSIONS.get(active_stack, set())
        for blocked_stack in blocked:
            blocked_kws = TECH_STACK_KEYWORDS.get(blocked_stack, set())
            if any(kw in text for kw in blocked_kws):
                return False

        return True

    # ------------------------------------------------------------------
    # [CHANGED] Scoring — simulation_requested now gives a strong boost
    # ------------------------------------------------------------------

    def _score_candidate(
        self,
        entry: CatalogEntry,
        family_weights: dict[str, int],
        explicit_stacks: set[str],
        simulation_requested: bool = False,
        family_mode: FamilyMode = FamilyMode.UNKNOWN,
        sjt_requested: bool = False,
    ) -> int:
        score = family_weights.get(entry.test_type, 0) * 10

        text = normalize_text(
            f"{entry.name} {entry.description or ''} {entry.category or ''}"
        )

        # ── [CHANGED] Simulation handling ────────────────────────────
        # Previously: always penalise simulation terms (-150).
        # Now: when simulation_requested=True we BOOST instead of penalise.
        #      The boost is large (+250) so simulation entries always surface
        #      above generic K/A/P entries for explicit simulation queries.
        if simulation_requested:
            # Boost entries that are simulation/contact-center products
            if self._is_simulation_entry(entry):
                score += 250   # strong positive boost — these should rank first

            # Extra named-product bonus for the primary simulation assessment
            if "contact center" in text or "contact centre" in text:
                score += 100   # additional bump for the canonical CC product

        else:
            # Default suppression: penalise noisy simulation entries when
            # the user has NOT asked for simulations (unchanged behaviour)
            for noise_term in NOISY_SIMULATION_TERMS:
                if noise_term in text:
                    score -= 150
                    break

        # ── Noise: SJT suppression for technical roles ───────────────
        if not sjt_requested and self._is_sjt_entry(entry):
            score -= 200

        # ── Cross-family purity penalties ────────────────────────────
        # [NOTE] When simulation_requested we relax family-mode penalties
        # so that simulation entries from any test type can surface.
        if not simulation_requested:
            if family_mode == FamilyMode.P_ONLY:
                if entry.test_type == "K":
                    score -= 500
                elif entry.test_type == "A":
                    score -= 120

            elif family_mode == FamilyMode.A_ONLY:
                if entry.test_type == "K":
                    score -= 200
                elif entry.test_type == "P":
                    score -= 60

            elif family_mode == FamilyMode.K_ONLY:
                if entry.test_type == "P":
                    score -= 80
                elif entry.test_type == "A":
                    score -= 40

        # ── K scoring ────────────────────────────────────────────────
        if entry.test_type == "K":
            for stack in explicit_stacks:
                allowed = TECH_STACK_KEYWORDS.get(stack, set())
                if any(kw in text for kw in allowed):
                    score += 70

        # ── A scoring ────────────────────────────────────────────────
        if entry.test_type == "A":
            reasoning_terms = {
                "reasoning", "cognitive", "problem",
                "analytical", "logic", "ability", "aptitude",
            }
            hits = sum(1 for term in reasoning_terms if term in text)
            score += hits * 14

        # ── P scoring ────────────────────────────────────────────────
        if entry.test_type == "P":
            personality_terms = {
                "communication", "collaboration", "teamwork",
                "stakeholder", "leadership", "interpersonal",
                "motivat", "behaviour", "behavior",
            }
            hits = sum(1 for term in personality_terms if term in text)
            score += hits * 18

            if any(kw in text for kw in PREFERRED_P_TYPES):
                score += 40

            # Only suppress simulation inside P scoring when NOT requested
            if not simulation_requested and "simulation" in text:
                score -= 80

        return score

    # ------------------------------------------------------------------
    # [CHANGED] Candidate collection — adds targeted simulation retrieval
    # ------------------------------------------------------------------

    def _collect_candidates(
        self,
        query: str,
        seniority: str | None,
        family_weights: dict[str, int],
        explicit_stacks: set[str],
        simulation_requested: bool = False,
        family_mode: FamilyMode = FamilyMode.UNKNOWN,
        sjt_requested: bool = False,
    ) -> list[CatalogEntry]:
        """
        Retrieve candidates from the catalog with mode-aware filtering.

        [CHANGED] When simulation_requested=True:
          - Run dedicated retrieval passes for each simulation query string.
          - Skip the technical-match relevance filter for simulation entries
            so they are never accidentally suppressed.
          - Still run the standard broad retrieval passes to ensure non-
            simulation assessments also appear for mixed queries.
        """
        merged: dict[str, CatalogEntry] = {}

        # ── [NEW] Simulation-first retrieval block ────────────────────
        # When the user has explicitly asked for simulation/contact-center
        # content, fetch it directly using targeted query strings BEFORE
        # the standard family-mode retrieval. This guarantees simulation
        # entries enter the candidate pool regardless of the family mode.
        if simulation_requested:
            for sim_query in SIMULATION_RETRIEVAL_QUERIES:
                for e in self.catalog.retrieve(sim_query, top_k=10, job_level=seniority):
                    merged[e.name] = e

        # ── Standard family-mode retrieval (unchanged logic) ──────────
        if family_mode == FamilyMode.P_ONLY:
            for e in self.catalog.retrieve(query, top_k=20, test_type="P", job_level=seniority):
                merged[e.name] = e
            for e in self.catalog.retrieve(query, top_k=5, test_type="A", job_level=seniority):
                merged[e.name] = e

        elif family_mode == FamilyMode.A_ONLY:
            for e in self.catalog.retrieve(query, top_k=15, test_type="A", job_level=seniority):
                merged[e.name] = e
            for e in self.catalog.retrieve(query, top_k=5, test_type="P", job_level=seniority):
                merged[e.name] = e

        elif family_mode == FamilyMode.K_ONLY:
            for e in self.catalog.retrieve(query, top_k=15, test_type="K", job_level=seniority):
                if self._is_relevant_technical_match(e, explicit_stacks):
                    merged[e.name] = e
            for e in self.catalog.retrieve(query, top_k=10, job_level=seniority):
                if self._is_relevant_technical_match(e, explicit_stacks):
                    merged[e.name] = e

        else:
            # MIXED or UNKNOWN
            for e in self.catalog.retrieve(query, top_k=20, job_level=seniority):
                if self._is_relevant_technical_match(e, explicit_stacks):
                    merged[e.name] = e
            for family in sorted(family_weights.keys()):
                for e in self.catalog.retrieve(
                    query, top_k=10, test_type=family, job_level=seniority
                ):
                    if self._is_relevant_technical_match(e, explicit_stacks):
                        merged[e.name] = e

        ranked = sorted(
            merged.values(),
            key=lambda e: (
                -self._score_candidate(
                    e, family_weights, explicit_stacks,
                    simulation_requested, family_mode, sjt_requested,
                ),
                e.test_type,
                e.name,
            ),
        )

        return ranked

    # ------------------------------------------------------------------
    # Noise helpers
    # ------------------------------------------------------------------

    def _is_noisy_entry(self, entry: CatalogEntry) -> bool:
        text = normalize_text(
            f"{entry.name} {entry.description or ''} {entry.category or ''}"
        )
        return any(noise in text for noise in NOISY_SIMULATION_TERMS)

    # ------------------------------------------------------------------
    # [CHANGED] Diversification — simulation entries get priority slots
    # ------------------------------------------------------------------

    def _diversify_recommendations(
        self,
        candidates: list[CatalogEntry],
        family_weights: dict[str, int],
        explicit_families: set[str],
        simulation_requested: bool = False,
        max_results: int = 4,
        family_mode: FamilyMode = FamilyMode.UNKNOWN,
        sjt_requested: bool = False,
    ) -> list[CatalogEntry]:
        """
        Select the final shortlist.

        [CHANGED] When simulation_requested=True:
          - Simulation entries skip the noise filter entirely.
          - Simulation entries are seeded into the shortlist first (up to 2
            slots) before normal diversification fills the remainder.
          - This guarantees Contact Center Simulation etc. appear at the top.
        """
        if not candidates:
            return []

        # ── [CHANGED] Noise filter: simulation entries always kept when requested
        filtered = [
            e for e in candidates
            if simulation_requested
            or not self._is_noisy_entry(e)
        ] or candidates

        # Hard-filter: remove SJT entries (unless explicitly requested)
        if not sjt_requested:
            without_sjt = [e for e in filtered if not self._is_sjt_entry(e)]
            if without_sjt:
                filtered = without_sjt

        selected: list[CatalogEntry] = []
        used: set[str] = set()

        def _pick_one(family: str) -> CatalogEntry | None:
            for e in filtered:
                if e.name not in used and e.test_type == family:
                    return e
            return None

        def _fill_from(family: str, cap: int) -> None:
            for e in filtered:
                if len(selected) >= cap:
                    break
                if e.name not in used and e.test_type == family:
                    selected.append(e)
                    used.add(e.name)

        def _fill_remainder(cap: int = max_results) -> None:
            for e in filtered:
                if len(selected) >= cap:
                    break
                if e.name not in used:
                    selected.append(e)
                    used.add(e.name)

        # ── [NEW] Simulation priority seeding ────────────────────────
        # When the user explicitly requests simulations, guarantee that up
        # to 2 simulation entries appear at the very front of the shortlist
        # before any family-mode logic runs. This is the key mechanism that
        # ensures "Contact Center Simulation" surfaces for explicit requests.
        if simulation_requested:
            sim_slots = min(2, max_results)
            for e in filtered:
                if len(selected) >= sim_slots:
                    break
                if e.name not in used and self._is_simulation_entry(e):
                    selected.append(e)
                    used.add(e.name)
            # Fill remaining slots with the best-scored non-simulation entries
            _fill_remainder(max_results)
            return selected[:max_results]

        # ── Standard diversification (unchanged) ─────────────────────
        if family_mode == FamilyMode.MIXED:
            families_present = explicit_families & {"K", "A", "P"}
            if not families_present:
                families_present = set(family_weights.keys()) & {"K", "A", "P"}

            for fam in ("K", "A", "P"):
                if fam in families_present:
                    entry = _pick_one(fam)
                    if entry:
                        selected.append(entry)
                        used.add(entry.name)

            _fill_remainder()

        elif family_mode == FamilyMode.P_ONLY:
            _fill_from("P", max_results)
            if len(selected) < max_results:
                _fill_from("A", max_results)

        elif family_mode == FamilyMode.A_ONLY:
            _fill_from("A", max_results - 1)
            if len(selected) < max_results:
                entry = _pick_one("P")
                if entry:
                    selected.append(entry)
                    used.add(entry.name)
            _fill_remainder()

        else:
            dominant = "K"
            if family_weights:
                dominant = max(family_weights, key=family_weights.__getitem__)

            _fill_from(dominant, max_results - 1)

            if len(selected) < max_results and explicit_families:
                secondary_families = explicit_families - {dominant}
                for e in filtered:
                    if len(selected) >= max_results:
                        break
                    if e.name not in used and e.test_type in secondary_families:
                        selected.append(e)
                        used.add(e.name)
                        break

            _fill_remainder()

        return selected[:max_results]

    # ------------------------------------------------------------------
    # Recommendation handler
    # ------------------------------------------------------------------

    def _handle_recommendation(
        self,
        context,
        latest_message: str,
    ) -> ChatResponse:
        query = context.build_query()

        # ── CLAUSE-AWARE stack extraction ────────────────────────────
        explicit_stacks = self._extract_latest_explicit_stacks(latest_message)

        # ── Signals from the current turn ────────────────────────────
        latest_weights, _ = self._detect_role_signals(latest_message)
        latest_explicit = self._explicit_families(latest_message)

        family_mode = self._derive_family_mode(latest_explicit, latest_weights)

        # ── Stack fallback ────────────────────────────────────────────
        if not explicit_stacks:
            _, query_stacks = self._detect_role_signals(query)
            explicit_stacks = query_stacks

        # ── Rewrite query: strip stale stack keywords ─────────────────
        if explicit_stacks:
            query = self._rewrite_query_for_latest_stack(query, explicit_stacks)

        # ── Family weights ────────────────────────────────────────────
        if latest_weights:
            family_weights = latest_weights
        else:
            family_weights, _ = self._detect_role_signals(query)

        if not family_weights:
            family_weights = {"K": 2}

        # ── [CHANGED] Simulation / SJT check ─────────────────────────
        # Check both the latest message AND the full query so that simulation
        # context from earlier turns is also captured.
        combined_text = f"{latest_message} {query}"
        simulation_requested = self._user_requested_simulations(combined_text)
        sjt_requested = self._user_requested_sjt(combined_text)

        # ── [NEW] When simulation is explicitly requested, override the
        # family_mode to UNKNOWN so that cross-family penalties are relaxed
        # and simulation entries from any test_type can surface freely.
        if simulation_requested:
            family_mode = FamilyMode.UNKNOWN

        # ── Explicit families for diversification ─────────────────────
        if latest_explicit:
            explicit_families = latest_explicit
        else:
            explicit_families = self._explicit_families(combined_text)

        # ── Collect + score ───────────────────────────────────────────
        candidates = self._collect_candidates(
            query=query,
            seniority=context.seniority,
            family_weights=family_weights,
            explicit_stacks=explicit_stacks,
            simulation_requested=simulation_requested,
            family_mode=family_mode,
            sjt_requested=sjt_requested,
        )

        if not candidates:
            return ChatResponse(
                reply=(
                    "I could not find strong SHL catalog matches for this role. "
                    "Please provide more detail."
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        final_entries = self._diversify_recommendations(
            candidates,
            family_weights,
            explicit_families=explicit_families,
            simulation_requested=simulation_requested,
            max_results=4,
            family_mode=family_mode,
            sjt_requested=sjt_requested,
        )

        recommendations: list[Recommendation] = []
        for entry in final_entries:
            if not self.catalog.is_catalog_url(entry.url):
                continue
            recommendations.append(
                Recommendation(
                    name=entry.name,
                    url=entry.url,
                    test_type=entry.test_type,
                )
            )

        reply = (
            "Updated the shortlist based on your refined requirements."
            if context.is_refinement
            else "Based on the role requirements, here are recommended SHL assessments."
        )

        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=True,
        )

    # ------------------------------------------------------------------
    # Compare handler
    # ------------------------------------------------------------------

    def _handle_compare(self, assessment_names: list[str]) -> ChatResponse:
        resolved: list[CatalogEntry] = []

        for name in assessment_names[:2]:
            entry = self._resolve_assessment_name(name)
            if entry:
                resolved.append(entry)

        if len(resolved) < 2:
            unresolved = assessment_names[len(resolved):]
            hint = (
                f" Could not resolve: {', '.join(unresolved)}."
                if unresolved
                else ""
            )
            return ChatResponse(
                reply=(
                    "Please provide the names of two SHL assessments to compare "
                    f"(e.g. OPQ and GSA, or Verify Interactive G+ and General Ability Screen).{hint}"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        first, second = resolved

        first_focus = (
            first.description or first.category or "specific workplace capabilities"
        )
        second_focus = (
            second.description or second.category or "specific workplace capabilities"
        )

        reply = (
            f"{first.name} (type {first.test_type}) focuses on {first_focus}. "
            f"{second.name} (type {second.test_type}) focuses on {second_focus}. "
            "Choose based on the capabilities most important for the role."
        )

        return ChatResponse(
            reply=reply,
            recommendations=[],
            end_of_conversation=False,
        )