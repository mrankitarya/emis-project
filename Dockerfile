FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# ── STEP 1: Core numeric deps ─────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "scipy==1.13.0"

# ── STEP 2: spaCy sub-deps in order (fixes resolution-too-deep) ───────────────
RUN pip install --no-cache-dir \
    "murmurhash==1.0.10" \
    "cymem==2.0.8" \
    "preshed==3.0.9" \
    "blis==0.7.11" \
    "catalogue==2.0.10" \
    "srsly==2.4.8" \
    "wasabi==1.1.3" \
    "thinc==8.2.4"

# ── STEP 3: pydantic pinned together (2.7.1 requires core==2.18.2 exactly) ────
RUN pip install --no-cache-dir \
    "pydantic==2.7.1" \
    "pydantic-core==2.18.2"

# ── STEP 4: ML models ─────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "scikit-learn==1.4.2" \
    "xgboost==2.0.3"

# ── STEP 5: Remaining requirements ────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
