"""FastAPI entrypoint for the SHL Assessment Recommender."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

# ====================== CORS MIDDLEWARE (Critical for Streamlit) ======================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Allow all origins (safe for this assignment)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize shared resources once at startup
try:
    catalog = CatalogStore()
    agent = SHLRecommenderAgent(catalog=catalog)
except Exception as exc:
    logger.exception("Failed to initialize application: %s", exc)
    catalog = None
    agent = None


@app.get("/health")
def health() -> dict[str, str]:
    """Readiness endpoint required by the assignment."""
    return {"status": "ok"}


@app.get("/")
def root():
    """Friendly root endpoint"""
    return {
        "message": "SHL Assessment Recommender API is running successfully!",
        "health": "/health",
        "docs": "/docs",
        "chat_endpoint": "/chat"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Process the full stateless conversation history."""
    if agent is None:
        return ChatResponse(
            reply="The service is temporarily unavailable. Please try again shortly.",
            recommendations=[],
            end_of_conversation=False,
        )

    try:
        return agent.chat(request.messages)
    except Exception as exc:
        logger.exception("Unhandled error while processing /chat: %s", exc)
        return ChatResponse(
            reply="I encountered an internal error while processing your request. Please provide the role details again.",
            recommendations=[],
            end_of_conversation=False,
        )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception) -> JSONResponse:
    """Catch unexpected framework-level exceptions."""
    logger.exception("Unhandled application exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "reply": "I encountered an internal error while processing your request.",
            "recommendations": [],
            "end_of_conversation": False,
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
