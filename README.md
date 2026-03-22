# EMIS — Encrypted Multi-Modal Intelligence System

[![CI/CD](https://github.com/mrankitarya/emis-project/actions/workflows/ci.yml/badge.svg)](https://github.com/mrankitarya/emis-project/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A scalable, production-ready **multi-agent intelligent system** that:

- Accepts **encrypted** text + image inputs
- Applies a **custom cipher** (XOR → byte-shift → block-scramble → Base64)
- Runs **NLP analysis** — sentiment, keyword extraction, entity recognition, risk detection
- Runs **Computer Vision** anomaly detection — PyTorch autoencoder + OpenCV fallback
- Produces a **unified risk score** via XGBoost / Random Forest / Logistic Regression
- Exposes a full **FastAPI REST API** with Swagger docs at `/docs`
- Stores all results in **MongoDB Atlas** (falls back to in-memory for local dev)
- Displays live insights on a **real-time dashboard** (Chart.js)
- Supports **async processing** via Celery + Redis
- Deploys via **Docker**, **Kubernetes**, and **CI/CD** (GitHub Actions → Render)

> **GitHub:** https://github.com/mrankitarya/emis-project

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   ENCRYPTED INPUT LAYER                  │
│          Encrypted Text  +  Encrypted Image (b64)        │
└─────────────────────┬────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│               CUSTOM DECRYPTION ENGINE                   │
│   Base64 decode → unscramble → unshift-13 → XOR(key)   │
└─────────────┬────────────────────────┬───────────────────┘
              │                        │
              ▼                        ▼
┌─────────────────────┐   ┌───────────────────────────┐
│      NLP AGENT      │   │      VISION AGENT (CV)    │
│  • Sentiment        │   │  • PyTorch Autoencoder    │
│  • Entity extract   │   │  • OpenCV anomaly scan    │
│  • Risk keywords    │   │  • Defect detection       │
└─────────────┬───────┘   └──────────────┬────────────┘
              │                          │
              └────────────┬─────────────┘
                           ▼
            ┌──────────────────────────┐
            │     RISK SCORING MODEL   │
            │  XGBoost / RF / LR       │
            │  → Unified risk score    │
            └──────────┬───────────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
  ┌───────────┐ ┌───────────┐ ┌──────────────┐
  │  FastAPI  │ │  MongoDB  │ │  Dashboard   │
  │   REST    │ │   Atlas   │ │  (HTML+JS)   │
  └───────────┘ └───────────┘ └──────────────┘
         │
  ┌──────┴──────┐
  │ Celery+Redis│  ← async bonus
  └─────────────┘
```

### Custom Encryption Pipeline

```
ENCRYPT:  plaintext → XOR(SHA-256 key) → shift+13 → block-scramble → Base64
DECRYPT:  Base64 → unscramble → unshift-13 → XOR(SHA-256 key) → plaintext
```

| Stage | What it does |
|---|---|
| **XOR** | Every byte XOR'd with repeating SHA-256 derived key |
| **Shift** | Caesar-style byte shift by 13 (mod 256) |
| **Scramble** | Split into 4-byte blocks → reverse each block → reverse block order |
| **Base64** | URL-safe Base64 encoding with 7-byte magic header + length prefix |

Does **not** rely solely on AES/Fernet — fully custom, fully reversible pipeline. Works on both text strings and raw image bytes.

---

## Quickstart (Local)

### 1. Clone

```bash
git clone https://github.com/mrankitarya/emis-project.git
cd emis-project
```

### 2. Set environment variables

```bash
cp .env.example .env
```

Edit `.env`:
```env
EMIS_SECRET_KEY=your-strong-secret-key-here
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/emis
REDIS_URL=redis://localhost:6379/0
```

> `MONGO_URI` and `REDIS_URL` are optional — the system uses in-memory fallbacks automatically if not set.

### 3a. Run with Docker Compose (recommended)

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Dashboard | http://localhost:3000 |
| Celery Flower monitor | http://localhost:5555 |

### 3b. Run without Docker

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Start the API
uvicorn main:app --reload --port 8000

# Open dashboard in browser
open dashboard/index.html       # macOS
# or just open the file in Chrome/Firefox
```

---

## API Reference

**Base URL:**
- Local: `http://localhost:8000`
- Deployed: `https://emis-api.onrender.com`

Interactive docs available at `/docs` (Swagger UI) and `/redoc`.

---

### Health

```http
GET /
GET /health
```
```json
{ "status": "ok" }
```

---

### Encrypt text

```http
POST /api/encrypt/text
Content-Type: application/json

{ "plaintext": "Suspicious login attempt detected from unknown IP." }
```
```json
{ "ciphertext": "yv4BAAAALg...base64..." }
```

---

### Decrypt text

```http
POST /api/decrypt/text
Content-Type: application/json

{ "plaintext": "yv4BAAAALg...base64..." }
```
```json
{ "plaintext": "Suspicious login attempt detected from unknown IP." }
```

---

### Encrypt image

```http
POST /api/encrypt/image
Content-Type: multipart/form-data

file=@image.jpg
```
```json
{
  "encrypted_image_b64": "yv4BAAAA...base64...",
  "filename": "image.jpg",
  "original_size_bytes": 48210,
  "encrypted_size_bytes": 64284
}
```

---

### Decrypt image (JSON response)

Returns the decrypted image as a Base64 string + a ready-to-use `data_url`.

```http
POST /api/decrypt/image
Content-Type: application/json

{
  "encrypted_image_b64": "yv4BAAAA...base64...",
  "filename": "my_image.jpg"
}
```
```json
{
  "filename": "my_image.jpg",
  "decrypted_image_b64": "/9j/4AAQSkZJRg...",
  "mime_type": "image/jpeg",
  "size_bytes": 48210,
  "data_url": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "message": "Image decrypted successfully. Use 'data_url' directly as an <img> src."
}
```

---

### Decrypt image (file download)

Returns the raw decrypted binary file — browser will save or display it directly.

```http
POST /api/decrypt/image/download
Content-Type: application/json

{
  "encrypted_image_b64": "yv4BAAAA...base64...",
  "filename": "my_image.jpg"
}
```

Returns: binary file response (`image/jpeg`, `image/png`, `image/gif`, or `image/webp`).

---

### Analyse — main endpoint

Full pipeline: decrypt → NLP + CV → risk score → save to DB → return result.

```http
POST /api/analyse
Content-Type: application/json

{
  "encrypted_text": "yv4BAAAALg...base64...",
  "encrypted_image_b64": null
}
```

```json
{
  "record_id": "3f2a1b4c-...",
  "risk_score": 0.8731,
  "risk_level": "CRITICAL",
  "explanation": "Risk level is CRITICAL (score=0.87). Primary driver: nlp_risk_score. NLP risk=0.89, CV anomaly=0.00. Detected 3 risk keyword(s): ransomware, backdoor, exploit.",
  "nlp": {
    "sentiment_label": "NEGATIVE",
    "sentiment_score": 0.9987,
    "keywords": ["login attempt", "server node"],
    "entities": [{ "text": "CVE-2024-1234", "label": "ORG" }],
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
    }
  }
}
```

**Risk levels:**

| Score | Level |
|---|---|
| 0.00 – 0.24 | LOW |
| 0.25 – 0.49 | MEDIUM |
| 0.50 – 0.74 | HIGH |
| 0.75 – 1.00 | CRITICAL |

---

### Async analyse (Celery + Redis)

```http
POST /api/analyse/async
```
```json
{ "task_id": "abc-123", "status": "queued", "message": "Poll /api/task/abc-123 for result." }
```

```http
GET /api/task/{task_id}
```
```json
{ "task_id": "abc-123", "status": "SUCCESS", "result": { "risk_score": 0.87, ... } }
```

Requires `REDIS_URL` environment variable. Falls back to `/api/analyse` (synchronous) if Redis is not configured.

---

### Upload and analyse (convenience)

Send encrypted text + raw image file together. Server encrypts the image automatically.

```http
POST /api/upload-analyse
Content-Type: multipart/form-data

encrypted_text=yv4BAAAALg...
image=@photo.jpg
```

---

### Dashboard data

```http
GET /api/records?limit=20     # recent records
GET /api/stats                # aggregate stats + risk distribution
```

---

## Running Tests

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_system.py::TestEncryption::test_encrypt_decrypt_roundtrip  PASSED
tests/test_system.py::TestEncryption::test_encrypt_bytes_roundtrip    PASSED
tests/test_system.py::TestEncryption::test_wrong_key_raises           PASSED
tests/test_system.py::TestEncryption::test_empty_key_raises           PASSED
tests/test_system.py::TestEncryption::test_ciphertext_is_base64_ascii PASSED
tests/test_system.py::TestNLPAgent::test_risk_keywords_detected       PASSED
tests/test_system.py::TestNLPAgent::test_clean_text_low_risk          PASSED
tests/test_system.py::TestRiskModel::test_high_nlp_high_cv_gives_high PASSED
tests/test_system.py::TestRiskModel::test_low_scores_give_low_risk    PASSED
tests/test_system.py::TestDatabase::test_save_and_retrieve            PASSED
tests/test_system.py::TestDatabase::test_stats_returns_dict           PASSED
```

---

## Deployment

### Render (backend API)

1. Connect `https://github.com/mrankitarya/emis-project` to [render.com](https://render.com)
2. Create a **Web Service** — Render will auto-detect `render.yaml`
3. Add environment variables in Render dashboard:

| Variable | Value |
|---|---|
| `EMIS_SECRET_KEY` | any strong random string |
| `MONGO_URI` | your MongoDB Atlas connection string |
| `REDIS_URL` | your Redis URL (optional) |

4. Copy the **Deploy Hook URL** from Render → add as `RENDER_DEPLOY_HOOK` in GitHub Secrets

### Docker (local / self-hosted)

```bash
# Build image
docker build -t emis:latest .

# Run full stack
docker-compose up --build

# Run only the API
docker run -p 8000:8000 \
  -e EMIS_SECRET_KEY=your-key \
  -e MONGO_URI=your-mongo-uri \
  emis:latest
```

> **Note on Docker build:** Heavy packages (torch, spacy, transformers, sklearn, xgboost) are installed in separate Dockerfile steps before `requirements.txt` runs. This prevents the `resolution-too-deep` pip error. Do not move these steps or merge them into a single `pip install`.

### Kubernetes

```bash
# Deploy everything
kubectl apply -f docker/k8s.yaml

# Check status
kubectl get pods -n emis

# Scale the API horizontally
kubectl scale deployment emis-api --replicas=4 -n emis
```

### CI/CD (GitHub Actions)

Triggers automatically on every push to `main`:

1. **Test** — runs full `pytest` suite
2. **Build** — builds Docker image, pushes to GitHub Container Registry (`ghcr.io`)
3. **Deploy** — triggers Render deploy hook

---

## Project Structure

```
emis-project/
├── app/
│   ├── api/
│   │   └── routes.py           # All FastAPI endpoints
│   ├── core/
│   │   ├── encryption.py       # Custom XOR+shift+scramble+Base64 cipher
│   │   ├── nlp_agent.py        # Sentiment + entity + risk keyword analysis
│   │   ├── vision_agent.py     # PyTorch autoencoder + OpenCV anomaly detection
│   │   └── database.py         # MongoDB Atlas + in-memory fallback
│   ├── ml/
│   │   └── risk_model.py       # XGBoost / RF / LR risk scorer
│   └── tasks/
│       ├── celery_app.py       # Celery + Redis configuration
│       └── process_tasks.py    # Async background task definitions
├── dashboard/
│   └── index.html              # Single-page real-time dashboard (Chart.js)
├── tests/
│   └── test_system.py          # Full test suite (encryption, NLP, ML, DB)
├── docker/
│   └── k8s.yaml                # Kubernetes manifests (API, worker, Redis)
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD pipeline
├── main.py                     # Uvicorn entry point
├── Dockerfile                  # Multi-step build (prevents pip conflicts)
├── docker-compose.yml          # Local full-stack setup
├── render.yaml                 # Render.com deployment config
├── requirements.txt            # Lightweight packages only (heavy ones in Dockerfile)
├── .env.example                # Environment variable template
└── README.md
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `EMIS_SECRET_KEY` | Yes | — | Encryption key for custom cipher |
| `MONGO_URI` | No | in-memory | MongoDB Atlas connection string |
| `MONGO_DB_NAME` | No | `emis` | MongoDB database name |
| `REDIS_URL` | No | disabled | Redis URL for Celery async queue |

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| API framework | FastAPI + Uvicorn | 0.111.0 |
| NLP | HuggingFace Transformers + spaCy | 4.40.0 / 3.7.4 |
| Computer Vision | PyTorch (CPU) + OpenCV | 2.2.2 / 4.9.0 |
| ML model | XGBoost / scikit-learn | 2.0.3 / 1.4.2 |
| Async queue | Celery + Redis | 5.4.0 / 5.0.4 |
| Database | MongoDB Atlas via Motor | 3.4.0 |
| Dashboard | HTML + Chart.js | — |
| Container | Docker + Docker Compose | — |
| Orchestration | Kubernetes | — |
| CI/CD | GitHub Actions → Render | — |

---

## Known Issues & Fixes Applied

| Issue | Fix |
|---|---|
| `resolution-too-deep` pip error | Heavy packages installed in separate Dockerfile steps before `requirements.txt` |
| `pydantic-core` version mismatch | Pinned `pydantic-core==2.18.2` to match `pydantic==2.7.1` exactly |
| `numpy` version conflict (torch vs spacy) | numpy pinned to `1.26.4` in Step 2, before spaCy or sklearn can override it |
| Missing image decrypt endpoint | Added `POST /api/decrypt/image` and `POST /api/decrypt/image/download` |

---

## License

MIT © 2026 [mrankitarya](https://github.com/mrankitarya)
