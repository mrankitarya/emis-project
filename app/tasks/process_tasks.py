"""
Async Celery tasks for heavy processing (bonus requirement).
"""
import asyncio
import base64
from app.tasks.celery_app import celery_app
from app.core.encryption import get_cipher
from app.core.nlp_agent import analyse_text
from app.core.vision_agent import analyse_image
from app.ml.risk_model import score_risk
from app.core import database


@celery_app.task(bind=True, max_retries=3, name="tasks.process_record")
def process_record_task(
    self,
    encrypted_text: str,
    encrypted_image_b64: str | None = None,
):
    """
    Full async pipeline:
      decrypt → NLP + CV in parallel → risk score → persist
    Returns the saved record dict.
    """
    try:
        cipher = get_cipher()

        # Decrypt text
        plain_text = cipher.decrypt(encrypted_text)

        # Decrypt image if provided
        image_bytes = None
        image_path = None
        if encrypted_image_b64:
            raw = cipher.decrypt_bytes(encrypted_image_b64)
            image_bytes = raw
            image_path = f"image_{self.request.id or 'sync'}.jpg"

        # NLP analysis
        nlp_result = analyse_text(plain_text)

        # CV analysis
        from app.core.vision_agent import CVResult
        cv_result = CVResult()
        if image_bytes:
            cv_result = analyse_image(image_bytes)

        # Risk scoring
        risk_result = score_risk(nlp_result, cv_result)

        # Persist (run async coroutine in sync context)
        record = asyncio.run(
            database.save_record(
                decrypted_text=plain_text,
                image_path=image_path,
                risk_score=risk_result.risk_score,
                risk_level=risk_result.risk_level,
                nlp_data=nlp_result.to_dict(),
                cv_data=cv_result.to_dict(),
                explanation=risk_result.explanation,
            )
        )
        return {
            "record_id": record["_id"],
            "risk_score": risk_result.risk_score,
            "risk_level": risk_result.risk_level,
            "explanation": risk_result.explanation,
        }

    except Exception as exc:
        raise self.retry(exc=exc, countdown=5)
