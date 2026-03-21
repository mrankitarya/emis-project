# EMIS — Encrypted Multi-Modal Intelligence System

[![CI/CD](https://github.com/YOUR_USERNAME/emis/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/emis/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A scalable, multi-agent intelligent system that:
- Accepts **encrypted** text + image inputs
- Applies a **custom cipher** (XOR → shift → scramble → Base64)
- Runs **NLP analysis** (sentiment, keyword, entity, risk detection)
- Runs **Computer Vision** anomaly detection (PyTorch autoencoder / OpenCV)
- Produces a **unified risk score** via XGBoost / Random Forest / Logistic Regression
- Exposes everything via a **FastAPI REST API**
- Stores results in **MongoDB Atlas**
- Displays insights on a **real-time dashboard**
- Supports **async processing** via Celery + Redis
- Deploys via **Docker**, **Kubernetes**, and **CI/CD** (GitHub Actions → Render)

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  ENCRYPTED INPUT LAYER               │
│         Encrypted Text  +  Encrypted Image           │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│              CUSTOM DECRYPTION ENGINE                │
│   Base64 decode → unscramble → unshift → XOR        │
└────────────┬──────────────────────┬──────────────────┘
             │                      │
             ▼                      ▼
┌────────────────────┐   ┌──────────────────────────┐
│     NLP AGENT      │   │      VISION AGENT (CV)   │
│  • Sentiment       │   │  • PyTorch Autoencoder   │
│  • Entity extract  │   │  • OpenCV anomaly scan   │
│  • Risk keywords   │   │  • Defect detection      │
└────────────┬───────┘   └──────────┬───────────────┘
             │                      │
             └──────────┬───────────┘
                        ▼
           ┌────────────────────────┐
           │    RISK SCORING MODEL  │
           │  XGBoost / RF / LR     │
           │  → Unified risk score  │
           └──────────┬─────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐
   │ FastAPI  │ │ MongoDB  │ │Dashboard │
   │   REST   │ │  Atlas   │ │  (HTML)  │
   └──────────┘ └──────────┘ └──────────┘
          │
   ┌──────┴──────┐
   │ Celery+Redis│  ← async bonus
   └─────────────┘
```

### Custom Encryption Pipeline

```
ENCRYPT:   plaintext → XOR(key) → shift+13 → block-scramble → Base64
DECRYPT:   Base64 → unscramble → unshift-13 → XOR(key) → plaintext
```

- **XOR**: every byte XOR'd with repeating SHA-256 derived key
- **Shift**: Caesar-style byte shift by 13 (mod 256)
- **Scramble**: split into 4-byte blocks, reverse each block, reverse block order
- **Encoding**: URL-safe Base64 with version header + length prefix for validation
- Does **not** rely solely on AES/Fernet — fully custom reversible pipeline

---

## Quickstart (Local)

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/emis.git
cd emis
```

### 2. Environment variables

```bash
cp .env.example .env
# Edit .env and set:
#   EMIS_SECRET_KEY=your-strong-secret
#   MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/emis
#   REDIS_URL=redis://localhost:6379/0   (optional, for async)
```

### 3a. Run with Docker Compose (recommended)

```bash
docker-compose up --build
```

| Service   | URL                        |
|-----------|----------------------------|
| API       | http://localhost:8000      |
| API Docs  | http://localhost:8000/docs |
| Dashboard | http://localhost:3000      |
| Flower    | http://localhost:5555      |

### 3b. Run without Docker

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Start API
uvicorn main:app --reload --port 8000

# Open dashboard
open dashboard/index.html
```

---

## API Reference

Base URL: `https://emis-api.onrender.com` (deployed) or `http://localhost:8000` (local)

### Health

```
GET /health
```
```json
{"status": "ok"}
```

---

### Encrypt text

```
POST /api/encrypt/text
Content-Type: application/json

{"plaintext": "Suspicious login attempt detected."}
```
```json
{"ciphertext": "yv4BAAAALg...base64...=="}
```

---

### Decrypt text

```
POST /api/decrypt/text
Content-Type: application/json

{"plaintext": "yv4BAAAALg...base64...=="}
```
```json
{"plaintext": "Suspicious login attempt detected."}
```

---

### Encrypt image

```
POST /api/encrypt/image
Content-Type: multipart/form-data

file=@image.jpg
```
```json
{
  "encrypted_image_b64": "yv4BAAAA...base64...",
  "filename": "image.jpg"
}
```

---

### Analyse (main endpoint)

```
POST /api/analyse
Content-Type: application/json

{
  "encrypted_text": "yv4BAAAALg...base64...",
  "encrypted_image_b64": null
}
```

**Response:**
```json
{
  "record_id": "3f2a1b4c-...",
  "risk_score": 0.8731,
  "risk_level": "CRITICAL",
  "explanation": "Risk level is CRITICAL (score=0.87). Primary driver: nlp risk score. NLP risk=0.89, CV anomaly=0.00. Detected 3 risk keyword(s): ransomware, backdoor, exploit.",
  "nlp": {
    "sentiment_label": "NEGATIVE",
    "sentiment_score": 0.9987,
    "keywords": ["login attempt", "server node"],
    "entities": [{"text": "CVE-2024-1234", "label": "ORG"}],
    "risk_keywords_found": ["ransomware", "backdoor", "exploit"],
    "risk_keyword_count": 3,
    "risk_keyword_density": 0.0857,
    "nlp_risk_score": 0.8931
  },
  "cv": {
    "anomaly_score": 0.0,
    "method": "numpy_fallback",
    "defects_detected": false
  },
  "risk": {
    "risk_score": 0.8731,
    "risk_level": "CRITICAL",
    "model_used": "XGBoost",
    "feature_importance": {
      "nlp_risk_score": 0.3241,
      "sentiment_negative": 0.2187,
      "risk_keyword_density": 0.1543
    },
    "explanation": "..."
  }
}
```

---

### Async analyse (Celery)

```
POST /api/analyse/async
→ {"task_id": "abc123", "status": "queued", "message": "..."}

GET /api/task/{task_id}
→ {"task_id": "abc123", "status": "SUCCESS", "result": {...}}
```

---

### Dashboard data

```
GET /api/records?limit=20   → recent records
GET /api/stats              → aggregate stats + distribution
```

---

## Running Tests

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_system.py::TestEncryption::test_encrypt_decrypt_roundtrip PASSED
tests/test_system.py::TestEncryption::test_encrypt_bytes_roundtrip PASSED
tests/test_system.py::TestEncryption::test_wrong_key_raises PASSED
tests/test_system.py::TestNLPAgent::test_risk_keywords_detected PASSED
tests/test_system.py::TestRiskModel::test_high_nlp_high_cv_gives_high_score PASSED
...
```

---

## Deployment

### Render (backend)

1. Connect your GitHub repo to [render.com](https://render.com)
2. Create a **Web Service** with:
   - Build command: `pip install -r requirements.txt && python -m spacy download en_core_web_sm`
   - Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. Add environment variables: `EMIS_SECRET_KEY`, `MONGO_URI`, `REDIS_URL`
4. Copy your **Deploy Hook URL** → add as `RENDER_DEPLOY_HOOK` GitHub secret

### Kubernetes

```bash
# Apply all manifests
kubectl apply -f docker/k8s.yaml

# Check pods
kubectl get pods -n emis

# Scale API
kubectl scale deployment emis-api --replicas=4 -n emis
```

### CI/CD (GitHub Actions)

Pipeline runs automatically on push to `main`:
1. **Test** – runs full pytest suite
2. **Build** – builds Docker image, pushes to GitHub Container Registry
3. **Deploy** – triggers Render deploy hook

---

## Project Structure

```
emis/
├── app/
│   ├── api/
│   │   └── routes.py          # FastAPI endpoints
│   ├── core/
│   │   ├── encryption.py      # Custom XOR+shift+scramble cipher
│   │   ├── nlp_agent.py       # Sentiment + entity + risk keywords
│   │   ├── vision_agent.py    # PyTorch autoencoder + OpenCV
│   │   └── database.py        # MongoDB Atlas / in-memory store
│   ├── ml/
│   │   └── risk_model.py      # XGBoost / RF / LR risk scorer
│   └── tasks/
│       ├── celery_app.py      # Celery + Redis configuration
│       └── process_tasks.py   # Async task definitions
├── dashboard/
│   └── index.html             # Single-page dashboard
├── tests/
│   └── test_system.py         # Full test suite
├── docker/
│   └── k8s.yaml               # Kubernetes manifests
├── .github/
│   └── workflows/ci.yml       # GitHub Actions CI/CD
├── main.py                    # Uvicorn entry point
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `EMIS_SECRET_KEY` | Yes | Encryption key for the custom cipher |
| `MONGO_URI` | No | MongoDB Atlas connection string (uses in-memory if unset) |
| `REDIS_URL` | No | Redis URL for Celery async tasks |

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| NLP | HuggingFace Transformers + spaCy |
| CV | PyTorch Autoencoder + OpenCV |
| ML | XGBoost / scikit-learn |
| Async | Celery + Redis |
| Database | MongoDB Atlas (Motor async driver) |
| Dashboard | Vanilla HTML + Chart.js |
| Container | Docker + Docker Compose |
| Orchestration | Kubernetes |
| CI/CD | GitHub Actions → Render |

---

## License

MIT
