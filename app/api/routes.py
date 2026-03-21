"""
EMIS – Encrypted Multi-Modal Intelligence System
FastAPI Application — COMPLETE with image decrypt endpoints
"""
from __future__ import annotations

import base64
import os
from typing import Annotated

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from app.core.encryption import get_cipher
from app.core.nlp_agent import analyse_text
from app.core.vision_agent import analyse_image, CVResult
from app.ml.risk_model import score_risk
from app.core import database

# ---------------------------------------------------------------------------
app = FastAPI(
    title="EMIS – Encrypted Multi-Modal Intelligence System",
    description="Processes encrypted text + images, runs NLP + CV analysis, returns risk scores.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class EncryptRequest(BaseModel):
    plaintext: str
    class Config:
        json_schema_extra = {
            "example": {"plaintext": "Suspicious login attempt detected from unknown IP."}
        }


class EncryptResponse(BaseModel):
    ciphertext: str


class AnalyseTextRequest(BaseModel):
    encrypted_text: str
    encrypted_image_b64: str | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "encrypted_text": "<base64-ciphertext-from-/api/encrypt/text>",
                "encrypted_image_b64": None,
            }
        }


class AnalyseResponse(BaseModel):
    record_id: str
    risk_score: float
    risk_level: str
    explanation: str
    nlp: dict
    cv: dict
    risk: dict


class AsyncJobResponse(BaseModel):
    task_id: str
    status: str
    message: str


# NEW: Schema for image decryption requests
class DecryptImageRequest(BaseModel):
    encrypted_image_b64: str
    filename: str | None = "decrypted_image.jpg"

    class Config:
        json_schema_extra = {
            "example": {
                "encrypted_image_b64": "yv4BAAAA...paste-output-from-/api/encrypt/image...",
                "filename": "my_image.jpg",
            }
        }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _use_celery() -> bool:
    return os.getenv("REDIS_URL", "") != ""


def _detect_image_type(raw_bytes: bytes) -> str:
    """Detect MIME type from magic bytes."""
    if raw_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw_bytes[:4] == b"GIF8":
        return "image/gif"
    if raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def root():
    return {"service": "EMIS", "status": "online", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}


# ======================================================================== #
#  ENCRYPTION GROUP                                                         #
# ======================================================================== #

@app.post("/api/encrypt/text", response_model=EncryptResponse, tags=["Encryption"])
async def encrypt_text(req: EncryptRequest):
    """
    Encrypt a plaintext string.

    Pipeline: UTF-8 bytes → XOR(key) → byte-shift +13 → block-scramble → Base64

    Returns a URL-safe Base64 ciphertext. Pass this to /api/analyse as
    'encrypted_text'.
    """
    cipher = get_cipher()
    return EncryptResponse(ciphertext=cipher.encrypt(req.plaintext))


@app.post("/api/decrypt/text", tags=["Encryption"])
async def decrypt_text(req: EncryptRequest):
    """
    Decrypt a ciphertext string back to plaintext.

    Pass the ciphertext (from /api/encrypt/text) in the 'plaintext' field.
    Returns 400 if the key is wrong or data is corrupted.
    """
    cipher = get_cipher()
    try:
        result = cipher.decrypt(req.plaintext)
        return {"plaintext": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Decryption failed: {e}")


@app.post("/api/encrypt/image", tags=["Encryption"])
async def encrypt_image(file: UploadFile = File(...)):
    """
    Upload a JPG / PNG / GIF / WEBP image file.

    The raw bytes are run through the same custom cipher as text:
    XOR → shift → scramble → Base64.

    Returns encrypted_image_b64 — pass this to /api/analyse as
    'encrypted_image_b64'.
    """
    cipher = get_cipher()
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    encrypted = cipher.encrypt_bytes(raw)
    return {
        "encrypted_image_b64": encrypted,
        "filename": file.filename,
        "original_size_bytes": len(raw),
        "encrypted_size_bytes": len(encrypted),
        "message": "Image encrypted. Pass 'encrypted_image_b64' to /api/analyse or /api/decrypt/image.",
    }


# ---- IMAGE DECRYPTION (THE MISSING PIECE) -------------------------------- #

@app.post("/api/decrypt/image", tags=["Encryption"])
async def decrypt_image_b64(req: DecryptImageRequest):
    """
    Decrypt an encrypted image (Base64 string) → returns decrypted image
    as a Base64 string inside JSON.

    Use this when you want to display the image in a browser / frontend
    by setting an <img> src to:
        data:image/jpeg;base64,{decrypted_image_b64}

    Pipeline (reverse of encrypt):
        Base64 decode → unscramble blocks → unshift bytes → XOR(key) → raw bytes
        → re-encode as standard Base64 for JSON transport
    """
    cipher = get_cipher()
    try:
        # Step 1: Decrypt back to original raw bytes
        raw_bytes = cipher.decrypt_bytes(req.encrypted_image_b64)

        # Step 2: Detect image type from magic bytes
        mime_type = _detect_image_type(raw_bytes)

        # Step 3: Re-encode as standard Base64 (not URL-safe) for JSON display
        recovered_b64 = base64.b64encode(raw_bytes).decode("ascii")

        return {
            "filename": req.filename,
            "decrypted_image_b64": recovered_b64,
            "mime_type": mime_type,
            "size_bytes": len(raw_bytes),
            "data_url": f"data:{mime_type};base64,{recovered_b64}",
            "message": (
                "Image decrypted successfully. "
                "Use 'data_url' directly as an <img> src, "
                "or use /api/decrypt/image/download to get the raw file."
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decryption failed: {e}")


@app.post("/api/decrypt/image/download", tags=["Encryption"])
async def decrypt_image_download(req: DecryptImageRequest):
    """
    Decrypt an encrypted image and return it as a downloadable binary file.

    The browser will either display the image inline or prompt a file save,
    depending on the image type. Use this when you need the actual file,
    not a Base64 string.

    Detected types: JPEG, PNG, GIF, WEBP (falls back to octet-stream).
    """
    cipher = get_cipher()
    try:
        raw_bytes = cipher.decrypt_bytes(req.encrypted_image_b64)
        filename = req.filename or "decrypted_image.jpg"
        mime_type = _detect_image_type(raw_bytes)

        return Response(
            content=raw_bytes,
            media_type=mime_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(raw_bytes)),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decryption failed: {e}")


# ======================================================================== #
#  ANALYSIS GROUP                                                           #
# ======================================================================== #

@app.post("/api/analyse", response_model=AnalyseResponse, tags=["Analysis"])
async def analyse(req: AnalyseTextRequest):
    """
    MAIN ENDPOINT — Full pipeline:

    1. Decrypt text (and image if provided)
    2. NLP agent → sentiment + keywords + risk keyword detection
    3. CV agent  → anomaly / defect detection (PyTorch autoencoder or OpenCV)
    4. Risk model → XGBoost / RF / LR → unified 0-1 risk score
    5. Persist to MongoDB Atlas (or in-memory store)
    6. Return full analysis result

    Both inputs must be pre-encrypted using /api/encrypt/text and
    /api/encrypt/image respectively.
    """
    cipher = get_cipher()

    # -- Decrypt text --------------------------------------------------------
    try:
        plain_text = cipher.decrypt(req.encrypted_text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Text decryption failed: {e}")

    # -- Decrypt image (optional) --------------------------------------------
    image_bytes = None
    image_path = None
    if req.encrypted_image_b64:
        try:
            image_bytes = cipher.decrypt_bytes(req.encrypted_image_b64)
            image_path = "uploaded_image.jpg"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image decryption failed: {e}")

    # -- NLP -----------------------------------------------------------------
    nlp_result = analyse_text(plain_text)

    # -- CV ------------------------------------------------------------------
    cv_result = CVResult()
    if image_bytes:
        cv_result = analyse_image(image_bytes)

    # -- Risk model ----------------------------------------------------------
    risk_result = score_risk(nlp_result, cv_result)

    # -- Persist -------------------------------------------------------------
    record = await database.save_record(
        decrypted_text=plain_text,
        image_path=image_path,
        risk_score=risk_result.risk_score,
        risk_level=risk_result.risk_level,
        nlp_data=nlp_result.to_dict(),
        cv_data=cv_result.to_dict(),
        explanation=risk_result.explanation,
    )

    return AnalyseResponse(
        record_id=record["_id"],
        risk_score=risk_result.risk_score,
        risk_level=risk_result.risk_level,
        explanation=risk_result.explanation,
        nlp=nlp_result.to_dict(),
        cv=cv_result.to_dict(),
        risk=risk_result.to_dict(),
    )


@app.post("/api/analyse/async", response_model=AsyncJobResponse, tags=["Analysis"])
async def analyse_async(req: AnalyseTextRequest):
    """
    Async version of /api/analyse — offloads work to a Celery worker via Redis.

    Returns immediately with a task_id. Poll /api/task/{task_id} for the result.
    Requires REDIS_URL environment variable to be set.
    """
    if not _use_celery():
        raise HTTPException(
            status_code=503,
            detail="Async processing requires REDIS_URL env var. Use /api/analyse instead.",
        )
    from app.tasks.process_tasks import process_record_task
    task = process_record_task.delay(req.encrypted_text, req.encrypted_image_b64)
    return AsyncJobResponse(
        task_id=task.id,
        status="queued",
        message=f"Task submitted. Poll /api/task/{task.id} for result.",
    )


@app.get("/api/task/{task_id}", tags=["Analysis"])
async def get_task_status(task_id: str):
    """Poll the status of an async Celery task by its task_id."""
    if not _use_celery():
        raise HTTPException(status_code=503, detail="Celery not configured.")
    from app.tasks.celery_app import celery_app as cel
    result = cel.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.ready() else None,
    }


@app.post("/api/upload-analyse", tags=["Analysis"])
async def upload_and_analyse(
    encrypted_text: Annotated[str, Form()],
    image: UploadFile = File(None),
):
    """
    Convenience endpoint for the dashboard form.

    Send encrypted text as a form field + raw image file together.
    The server encrypts the image automatically then runs the full pipeline.
    No need to pre-encrypt the image yourself.
    """
    cipher = get_cipher()
    encrypted_image_b64 = None
    if image and image.filename:
        raw = await image.read()
        if raw:
            encrypted_image_b64 = cipher.encrypt_bytes(raw)

    req = AnalyseTextRequest(
        encrypted_text=encrypted_text,
        encrypted_image_b64=encrypted_image_b64,
    )
    return await analyse(req)


# ======================================================================== #
#  DASHBOARD GROUP                                                          #
# ======================================================================== #

@app.get("/api/records", tags=["Dashboard"])
async def get_records(limit: int = 20):
    """
    Return the most recent analysis records for the dashboard table.
    Use ?limit=N to control how many records are returned (default 20, max 100).
    """
    limit = min(limit, 100)
    records = await database.get_recent(limit=limit)
    for r in records:
        r["id"] = r.pop("_id", r.get("id", ""))
    return {"records": records, "count": len(records)}


@app.get("/api/stats", tags=["Dashboard"])
async def get_stats():
    """
    Return aggregate statistics for the dashboard cards and charts:
    total records, avg/min/max risk score, and distribution by risk level.
    """
    return await database.get_stats()
