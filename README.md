# Customer Service Chatbot 

An **intelligent customer service chatbot** built with **FastAPI** (Python) and a lightweight **ML/NLP pipeline** (scikit‑learn + TF‑IDF). It supports:

- Intent classification (greet, track order, refund status, product info, return policy, etc.)
- FAQ retrieval with TF‑IDF similarity
- Simple entity extraction (order ID, email, phone) and slot filling
- Conversation logging to SQLite, with ticket creation for agent escalation
- React chat UI
- One‑command training and run

> Tested on macOS (Apple Silicon & Intel).

---

## 1) Prerequisites (macOS)

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Ensure Python 3.11+
brew install python@3.11

# (Optional) Node.js for the React frontend
brew install node
```

## 2) Backend setup

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# NLTK small setup (tokenizers/stopwords)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Install advanced NLP & PyTorch for Apple Silicon
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers scikit-learn spacy
python -m spacy download en_core_web_sm

# Download NLTK data
python -m nltk.downloader punkt stopwords
```

### Train models

```bash
python nlp/train.py
```

This will create `models/intent_clf.joblib`, `models/vectorizer.joblib`, and `models/faq_vectorizer.joblib`.

### Run API

```bash
uvicorn app:app --reload --port 8000
```

- Health check: `GET http://127.0.0.1:8000/health`
- Chat: `POST http://127.0.0.1:8000/chat` with JSON:
  ```json
  {
    "message": "I want to track my order 123-4567890-1234567",
    "user_id": "demo-user",
    "conversation_id": "conv-001"
  }
  ```

## 3) Frontend (React) setup

```bash
cd ../frontend
npm install
npm run dev
```

Open the printed local URL (usually http://localhost:5173).

## 4) Project structure

```
chatbot-customer-service-mac/
├── backend/
│   ├── app.py
│   ├── db.py
│   ├── schemas.py
│   ├── nlp/
│   │   ├── train.py
│   │   ├── intent_model.py
│   │   ├── faq.py
│   │   ├── ner.py
│   │   └── policy.py
│   ├── data/
│   │   ├── intents.json
│   │   └── faq.csv
│   ├── models/
│   ├── tests/
│   │   └── test_api.py
│   └── requirements.txt
└── frontend/
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── main.jsx
        └── App.jsx
```

## 5) Design notes

- ML: `LinearSVC` intent classifier + TF‑IDF makes training fast and portable (no big model downloads).
- FAQ: TF‑IDF cosine similarity with a confidence threshold.
- NER: simple regex extractors for common items (order IDs, email, phone). Easy to extend.
- Policy: tiny rule‑based dialog manager for slot filling + graceful fallbacks.
- Data: You can expand `data/intents.json` and `data/faq.csv` with your domain phrases.
- Logs & Tickets: SQLite for portability. Promote messages to tickets with `/escalate` intent or low confidence.

## 6) Testing

```bash
# with virtualenv active
pytest -q
```

## 7) Packaging & Deployment

- Dockerfile (optional) or run on any Python host
- Use a process manager (e.g., `gunicorn` + `uvicorn.workers.UvicornWorker` behind Nginx)
