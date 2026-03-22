# Base image
FROM python:3.11-slim

WORKDIR /app

# ── System deps for OpenCV + spaCy ──────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Upgrade pip ────────────────────────────────────────────────
RUN pip install --upgrade pip setuptools wheel

# ── Install PyTorch CPU first ─────────────────────────────────
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu

# ── Install numpy & scipy ──────────────────────────────────────
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.13.0

# ── Install spaCy & dependencies ───────────────────────────────
RUN pip install --no-cache-dir \
    murmurhash==1.0.10 \
    cymem==2.0.8 \
    preshed==3.0.9 \
    blis==0.7.11 \
    thinc==8.2.4 \
    spacy==3.7.4

# ── Copy requirements and install remaining packages ───────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Download spaCy English model ───────────────────────────────
RUN python -m spacy download en_core_web_sm || true

# ── Copy project files ────────────────────────────────────────
COPY . .

# ── Expose port and run ───────────────────────────────────────
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
