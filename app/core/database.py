"""
Database layer – MongoDB Atlas via Motor (async).
Falls back to an in-memory store when MONGO_URI is unset (for local dev/tests).
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any

MONGO_URI = os.getenv("MONGO_URI", "")
DB_NAME = os.getenv("MONGO_DB_NAME", "emis")
COLLECTION = "records"

# ---------------------------------------------------------------------------
# In-memory fallback store
# ---------------------------------------------------------------------------
_mem_store: list[dict] = []


def _new_id() -> str:
    return str(uuid.uuid4())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Async MongoDB helpers
# ---------------------------------------------------------------------------

_motor_client = None
_motor_db = None


async def _get_collection():
    global _motor_client, _motor_db
    if not MONGO_URI:
        return None
    if _motor_client is None:
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            _motor_client = AsyncIOMotorClient(MONGO_URI)
            _motor_db = _motor_client[DB_NAME]
        except Exception:
            return None
    return _motor_db[COLLECTION]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def save_record(
    decrypted_text: str,
    image_path: str | None,
    risk_score: float,
    risk_level: str,
    nlp_data: dict,
    cv_data: dict,
    explanation: str,
) -> dict[str, Any]:
    record = {
        "_id": _new_id(),
        "timestamp": _now(),
        "decrypted_text": decrypted_text,
        "image_path": image_path,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "nlp": nlp_data,
        "cv": cv_data,
        "explanation": explanation,
    }

    col = await _get_collection()
    if col is not None:
        try:
            await col.insert_one({**record})
        except Exception:
            _mem_store.append(record)
    else:
        _mem_store.append(record)

    return record


async def get_recent(limit: int = 20) -> list[dict]:
    col = await _get_collection()
    if col is not None:
        try:
            cursor = col.find({}, {"_id": 1, "timestamp": 1, "risk_score": 1,
                                   "risk_level": 1, "explanation": 1,
                                   "decrypted_text": 1}
                               ).sort("timestamp", -1).limit(limit)
            return await cursor.to_list(length=limit)
        except Exception:
            pass
    return list(reversed(_mem_store[-limit:]))


async def get_stats() -> dict[str, Any]:
    col = await _get_collection()
    records = []
    if col is not None:
        try:
            cursor = col.find({}, {"risk_score": 1, "risk_level": 1})
            records = await cursor.to_list(length=10_000)
        except Exception:
            records = _mem_store
    else:
        records = _mem_store

    if not records:
        return {"total": 0, "avg_risk": 0.0, "distribution": {}}

    scores = [r.get("risk_score", 0) for r in records]
    levels = [r.get("risk_level", "UNKNOWN") for r in records]
    distribution: dict[str, int] = {}
    for lv in levels:
        distribution[lv] = distribution.get(lv, 0) + 1

    return {
        "total": len(records),
        "avg_risk": round(sum(scores) / len(scores), 4),
        "max_risk": round(max(scores), 4),
        "min_risk": round(min(scores), 4),
        "distribution": distribution,
    }
