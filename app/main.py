"""FastAPI entrypoint for the SHL Assessment Recommender."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .agent import SHLRecommenderAgent
from .catalog import CatalogStore
from .schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Stateless conversational API for recommending SHL assessments.",
    version="1.0.0",
)

# Initialize shared resources once at startup for fast request handling.
try:
    catalog = CatalogStore()
    agent = SHLRecommenderAgent(catalog=catalog)
except Exception as exc:  # pragma: no cover - startup safeguard
    logger.exception("Failed to initialize application: %s", exc)
    catalog = None
    agent = None


@app.get("/health")
def health() -> dict[str, str]:
    """Readiness endpoint required by the assignment."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Process the full stateless conversation history."""
    if agent is None:
        return ChatResponse(
            reply=(
                "The service is temporarily unavailable. Please try again shortly."
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    try:
        return agent.chat(request.messages)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.exception("Unhandled error while processing /chat: %s", exc)
        return ChatResponse(
            reply=(
                "I encountered an internal error while processing your request. "
                "Please provide the role details again."
            ),
            recommendations=[],
            end_of_conversation=False,
        )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception) -> JSONResponse:
    """Catch unexpected framework-level exceptions and return a safe response."""
    logger.exception("Unhandled application exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "reply": (
                "I encountered an internal error while processing your request."
            ),
            "recommendations": [],
            "end_of_conversation": False,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)