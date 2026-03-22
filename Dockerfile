FROM python:3.11-slim

WORKDIR /app

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# ── Upgrade pip ───────────────────────────────────────────────────────────────
RUN pip install --upgrade pip setuptools wheel

# ── STEP 1: PyTorch CPU (install alone — biggest package, anchors the graph) ──
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu

# ── STEP 2: Pin numpy BEFORE spacy/sklearn can pull a different version ────────
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "scipy==1.13.0"

# ── STEP 3: spaCy sub-deps in strict order (fixes resolution-too-deep) ────────
RUN pip install --no-cache-dir \
    "murmurhash==1.0.10" \
    "cymem==2.0.8" \
    "preshed==3.0.9" \
    "blis==0.7.11" \
    "catalogue==2.0.10" \
    "srsly==2.4.8" \
    "wasabi==1.1.3" \
    "thinc==8.2.4" \
    "spacy==3.7.4"

# ── STEP 4: Transformers stack ────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "tokenizers==0.19.1" \
    "safetensors==0.4.3" \
    "huggingface-hub==0.23.0" \
    "transformers==4.40.0"

# ── STEP 5: ML models ─────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "scikit-learn==1.4.2" \
    "xgboost==2.0.3"

# ── STEP 6: Pydantic — pin BOTH together so versions never conflict ────────────
# pydantic==2.7.1 requires pydantic-core==2.18.2 exactly (NOT 2.18.3)
RUN pip install --no-cache-dir \
    "pydantic==2.7.1" \
    "pydantic-core==2.18.2"

# ── STEP 7: Remaining lightweight packages ────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# ── STEP 8: spaCy English model ───────────────────────────────────────────────
RUN python -m spacy download en_core_web_sm || true

# ── STEP 9: Copy source code last (maximises Docker layer cache) ──────────────
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
