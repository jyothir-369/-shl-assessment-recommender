# SHL Assessment Recommender

A stateless conversational API that recommends relevant SHL assessments from the SHL Product Catalog (restricted to Individual Test Solutions). The service interprets a multi-turn hiring conversation, asks clarifying questions when needed, and returns grounded recommendations with official SHL catalog URLs.

---

## Assignment Requirements Satisfied

This project implements all core requirements from the SHL AI Intern take-home assignment:

- Uses a local `catalog.json` built from SHL Individual Test Solutions only.
- Exposes:
  - `GET /health`
  - `POST /chat`
- Accepts the full conversation history on every request (stateless API).
- Supports:
  - Clarification of vague requests
  - Recommendations (1–10 assessments)
  - Refinement when requirements change
  - Comparison of named assessments
- Refuses:
  - General hiring advice
  - Legal and salary questions
  - Prompt-injection attempts
- Returns only catalog-backed assessment names and URLs.

---

## Project Structure

```text
project-root/
├── app/
│   ├── __init__.py
│   ├── agent.py
│   ├── catalog.py
│   ├── interpreter.py
│   ├── main.py
│   └── schemas.py
├── catalog.json
├── requirements.txt
├── README.md
└── tests/
    ├── test_api.py
    └── test_agent.py







Setup
1. Clone the Repository
git clone <your-repo-url>
cd project-root
2. Create a Virtual Environment
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
3. Install Dependencies
pip install -r requirements.txt
Environment Variables

No environment variables are required for the default implementation.

Optional (if you later integrate an LLM API):

OPENAI_API_KEY=...
GROQ_API_KEY=...
GEMINI_API_KEY=...

The current implementation works entirely with deterministic retrieval and rule-based logic.

Running the API Locally
uvicorn app.main:app --reload

The API will be available at:

http://127.0.0.1:8000
Swagger docs: http://127.0.0.1:8000/docs
API Endpoints
GET /health

Returns service readiness.

Response
{
  "status": "ok"
}
POST /chat

Processes the full conversation history and returns the next assistant response.

Request
{
  "messages": [
    {
      "role": "user",
      "content": "Hiring a Java developer who works with stakeholders"
    },
    {
      "role": "assistant",
      "content": "What is the seniority level for this role?"
    },
    {
      "role": "user",
      "content": "Mid-level, around 4 years"
    }
  ]
}
Response
{
  "reply": "Based on the role requirements, here are 5 SHL assessments to consider.",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
      "test_type": "K"
    },
    {
      "name": "Occupational Personality Questionnaire (OPQ32r)",
      "url": "https://www.shl.com/solutions/products/product-catalog/view/occupational-personality-questionnaire-opq32/",
      "test_type": "P"
    }
  ],
  "end_of_conversation": true
}
Testing the API
Using cURL
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I am hiring a Python developer"}
    ]
  }'
Running Unit Tests
pytest -v
Catalog Data

catalog.json contains structured entries from the SHL Product Catalog, restricted to Individual Test Solutions.

Each entry includes:

name
url
test_type
description
category
skills
job_levels

The CatalogStore loads this file at startup and validates that all URLs are SHL catalog URLs.

Retrieval and Recommendation Logic
1. Conversation Interpretation

interpreter.py extracts:

Role title
Seniority
Desired assessment types
Constraints
Mentioned assessments
Intent (clarify, recommend, refine, compare, off-topic)
2. Catalog Retrieval

catalog.py performs lightweight hybrid ranking using:

Exact name matching
Keyword overlap
Token-based lexical scoring
3. Agent Orchestration

agent.py decides whether to:

Ask a clarifying question
Recommend assessments
Refine a shortlist
Compare two assessments
Refuse out-of-scope requests

All recommendations are grounded in catalog.json.

Stateless API Design

The API stores no conversation state.

Every POST /chat request must include the complete conversation history, and the response is computed solely from that history.

This matches the assignment's required architecture and simplifies deployment.

Deployment

The application can be deployed to platforms such as:

Render
Railway
Fly.io
Hugging Face Spaces
Example Render Start Command
uvicorn app.main:app --host 0.0.0.0 --port $PORT
Design Principles
Deterministic and testable
No hallucinated recommendations
Only SHL catalog-backed URLs
Lightweight dependencies
Fast cold starts
Production-ready FastAPI structure
Future Improvements
Automated full-catalog scraper from SHL
TF-IDF or embedding-based ranking
Evaluation against provided sample conversations
Optional LLM-assisted explanation generation
License

This project was created for the SHL AI Intern take-home assignment.


