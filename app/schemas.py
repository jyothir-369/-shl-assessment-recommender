"""Pydantic schemas for the SHL Assessment Recommender API."""

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator


class Message(BaseModel):
    """A single message in the stateless conversation history."""

    role: Literal["user", "assistant"] = Field(
        ...,
        description="The sender of the message.",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="The text content of the message.",
    )


class ChatRequest(BaseModel):
    """Request payload for POST /chat containing the full conversation history."""

    messages: list[Message] = Field(
        ...,
        min_length=1,
        description="Complete conversation history in chronological order.",
    )


class Recommendation(BaseModel):
    """A recommended SHL assessment."""

    name: str = Field(
        ...,
        min_length=1,
        description="Assessment name from the SHL catalog.",
    )
    url: HttpUrl = Field(
        ...,
        description="Canonical URL from the SHL product catalog.",
    )
    test_type: str = Field(
        ...,
        min_length=1,
        description="Assessment type code (e.g., K, P, A).",
    )


class ChatResponse(BaseModel):
    """Response payload for POST /chat."""

    reply: str = Field(
        ...,
        min_length=1,
        description="The assistant's natural-language response.",
    )
    recommendations: list[Recommendation] = Field(
        default_factory=list,
        description=(
            "Recommended assessments. Must be empty while clarifying or refusing, "
            "or contain between 1 and 10 items when recommendations are provided."
        ),
    )
    end_of_conversation: bool = Field(
        ...,
        description="True only when the agent considers the task complete.",
    )

    @model_validator(mode="after")
    def validate_recommendation_count(self) -> "ChatResponse":
        """Ensure recommendations are either empty or contain 1 to 10 items."""
        count = len(self.recommendations)
        if count not in (0,) and not 1 <= count <= 10:
            raise ValueError(
                "recommendations must be empty or contain between 1 and 10 items."
            )
        return self