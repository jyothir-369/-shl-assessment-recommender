# SHL Conversational Assessment Recommender

A **stateless conversational AI** that helps hiring managers go from vague role descriptions to a grounded shortlist of **SHL Individual Test Solutions**.

This project was built as a submission for the **SHL AI Intern** take-home assignment.

---

## 🌐 Live Deployment

- **Frontend (Streamlit)**: [https://jyothir-shl-recommender.streamlit.app/](https://jyothir-shl-recommender.streamlit.app/)
- **Backend API (Render)**: [https://shl-recommender-api-3rd3.onrender.com](https://shl-recommender-api-3rd3.onrender.com)
- **API Documentation**: [https://shl-recommender-api-3rd3.onrender.com/docs](https://shl-recommender-api-3rd3.onrender.com/docs)

---

## Assignment Requirements Satisfied

This implementation fully addresses all core requirements of the SHL take-home assignment:

- **Catalog Handling**: Uses `catalog.json` containing only **Individual Test Solutions**.
- **API Endpoints**:
  - `GET /health`
  - `POST /chat` (stateless — accepts full conversation history)
- **Conversational Behaviors**:
  - Clarifies vague queries before recommending
  - Provides 1–10 relevant recommendations with official SHL URLs
  - Supports refinement (e.g., Java → Python, add personality)
  - Supports comparison between assessments
- **Safety**: Refuses off-topic, legal, salary, and prompt-injection attempts
- **Grounding**: All recommendations and URLs come strictly from the catalog

---

## Project Structure

```text
project-root/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── agent.py
│   ├── catalog.py
│   ├── interpreter.py
│   └── schemas.py
├── catalog.json
├── streamlit_app.py
├── requirements.txt
├── README.md
└── tests/

Setup & Running Locally
Bash# 1. Clone repo
git clone <your-repo-url>
cd project-root

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run FastAPI backend
uvicorn app.main:app --reload
API will be available at: http://127.0.0.1:8000
Swagger UI: http://127.0.0.1:8000/docs

API Usage
Health Check
Bashcurl https://shl-recommender-api-3rd3.onrender.com/health
Chat Endpoint
Bashcurl -X POST https://shl-recommender-api-3rd3.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I am hiring a mid-level Java developer"}
    ]
  }'

Key Features & Design Choices

Stateless Architecture: Full conversation history sent with every request.
Strong Clarification Logic: Avoids premature recommendations.
Hybrid Retrieval: Combines keyword matching and scoring for high relevance.
Balanced Recommendations: Intelligently mixes Knowledge (K), Ability (A), and Personality (P) tests when appropriate.
Refinement Support: Handles changes in role, level, or focus effectively.
Comparison Capability: Grounded comparison using catalog data only.
CORS Enabled: Supports frontend deployment on Streamlit Cloud.


Testing
Bashpytest -v
Recommended manual test prompts:

"I’m hiring a mid-level backend Java developer with 4 years experience..."
"Compare OPQ and GSA"
"Actually, change it to Python and add more personality assessments"


Deployment

Backend: Deployed on Render (FastAPI + Uvicorn)
Frontend: Deployed on Streamlit Community Cloud
Both services are publicly accessible.


License
This project was developed as part of the SHL AI Intern take-home assignment (2026).

Made with ❤️ for SHL Labs
text---

### Next Steps (Recommended)

1. Copy the above content into your `README.md`
2. Commit and push:

```bash
git add README.md
git commit -m "Update README with live links and improved documentation"
git push
