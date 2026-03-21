"""
EMIS – Encrypted Multi-Modal Intelligence System
FastAPI Application
"""
from __future__ import annotations

import base64
import os
from typing import Annotated

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    encrypted_image_b64: str | None = None   # optional encrypted image as b64

    class Config:
        json_schema_extra = {
            "example": {
                "encrypted_text": "<base64-ciphertext>",
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


# ---------------------------------------------------------------------------
# Utility – try Celery, fall back to sync
# ---------------------------------------------------------------------------

def _use_celery() -> bool:
    return os.getenv("REDIS_URL", "") != ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def root():
    return {"service": "EMIS", "status": "online", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}


# ---- Encryption helpers -------------------------------------------------- #

@app.post("/api/encrypt/text", response_model=EncryptResponse, tags=["Encryption"])
async def encrypt_text(req: EncryptRequest):
    """Encrypt a plaintext string using the custom cipher."""
    cipher = get_cipher()
    return EncryptResponse(ciphertext=cipher.encrypt(req.plaintext))


@app.post("/api/decrypt/text", tags=["Encryption"])
async def decrypt_text(req: EncryptRequest):
    """Decrypt a ciphertext string (pass ciphertext in 'plaintext' field)."""
    cipher = get_cipher()
    try:
        result = cipher.decrypt(req.plaintext)
        return {"plaintext": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Decryption failed: {e}")


@app.post("/api/encrypt/image", tags=["Encryption"])
async def encrypt_image(file: UploadFile = File(...)):
    """Accept an image file, return its encrypted base64 representation."""
    cipher = get_cipher()
    raw = await file.read()
    encrypted = cipher.encrypt_bytes(raw)
    return {"encrypted_image_b64": encrypted, "filename": file.filename}


# ---- Core analysis ------------------------------------------------------- #

@app.post("/api/analyse", response_model=AnalyseResponse, tags=["Analysis"])
async def analyse(req: AnalyseTextRequest):
    """
    Main endpoint: decrypt inputs → NLP + CV → risk score → persist.
    Runs synchronously (use /api/analyse/async for Celery offload).
    """
    cipher = get_cipher()

    # Decrypt text
    try:
        plain_text = cipher.decrypt(req.encrypted_text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Text decryption failed: {e}")

    # Decrypt image if provided
    image_bytes = None
    image_path = None
    if req.encrypted_image_b64:
        try:
            image_bytes = cipher.decrypt_bytes(req.encrypted_image_b64)
            image_path = "uploaded_image.jpg"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image decryption failed: {e}")

    # NLP
    nlp_result = analyse_text(plain_text)

    # CV
    cv_result = CVResult()
    if image_bytes:
        cv_result = analyse_image(image_bytes)

    # Risk
    risk_result = score_risk(nlp_result, cv_result)

    # Persist
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
    """Offload analysis to Celery worker (bonus async requirement)."""
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
        message="Task submitted. Poll /api/task/{task_id} for result.",
    )


@app.get("/api/task/{task_id}", tags=["Analysis"])
async def get_task_status(task_id: str):
    """Poll Celery task status."""
    if not _use_celery():
        raise HTTPException(status_code=503, detail="Celery not configured.")
    from app.tasks.celery_app import celery_app as cel
    result = cel.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.ready() else None,
    }


# ---- Dashboard data ------------------------------------------------------ #

@app.get("/api/records", tags=["Dashboard"])
async def get_records(limit: int = 20):
    """Return recent analysis records for the dashboard."""
    records = await database.get_recent(limit=limit)
    # Ensure _id is serialisable
    for r in records:
        r["id"] = r.pop("_id", r.get("id", ""))
    return {"records": records, "count": len(records)}


@app.get("/api/stats", tags=["Dashboard"])
async def get_stats():
    """Return aggregate stats for the dashboard."""
    return await database.get_stats()


# ---- Upload + full pipeline --------------------------------------------- #

@app.post("/api/upload-analyse", tags=["Analysis"])
async def upload_and_analyse(
    encrypted_text: Annotated[str, Form()],
    image: UploadFile = File(None),
):
    """
    Multipart form endpoint: accepts encrypted text + optional raw image file.
    Encrypts image automatically before processing.
    """
    cipher = get_cipher()
    encrypted_image_b64 = None
    if image:
        raw = await image.read()
        encrypted_image_b64 = cipher.encrypt_bytes(raw)

    req = AnalyseTextRequest(
        encrypted_text=encrypted_text,
        encrypted_image_b64=encrypted_image_b64,
    )
    return await analyse(req)
