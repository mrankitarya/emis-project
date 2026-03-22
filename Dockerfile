FROM python:3.11-slim

WORKDIR /app

# ── System packages needed by OpenCV + spaCy ─────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Upgrade pip, setuptools, wheel first ─────────────────────────────────────
RUN pip install --upgrade pip setuptools wheel

# ── STEP 1: PyTorch CPU only (biggest package — install ALONE first) ──────────
# CPU build avoids 2 GB CUDA download, keeps image under 3 GB
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu

# ── STEP 2: NumPy + SciPy pinned BEFORE anything else demands a version ───────
# Prevents numpy version conflicts between torch / spacy / sklearn
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "scipy==1.13.0"

# ── STEP 3: spaCy sub-deps in strict dependency order ─────────────────────────
# THIS is what fixes "resolution-too-deep" — pip never has to discover these
# Install order: murmurhash → cymem → preshed → blis → thinc → spacy
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

# ── STEP 4: Transformers sub-deps (install before requirements.txt) ───────────
RUN pip install --no-cache-dir \
    "tokenizers==0.19.1" \
    "safetensors==0.4.3" \
    "huggingface-hub==0.23.0" \
    "transformers==4.40.0"

# ── STEP 5: ML models ─────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "scikit-learn==1.4.2" \
    "xgboost==2.0.3"

# ── STEP 6: Remaining packages (graph is 80% solved — pip breezes through) ────
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# ── STEP 7: spaCy English model (|| true = don't fail on slow CDN) ────────────
RUN python -m spacy download en_core_web_sm || true

# ── STEP 8: Copy source code LAST (maximises Docker layer cache reuse) ─────────
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
