"""
Vision Agent - Lightweight version for Render free tier (512MB RAM)
Uses OpenCV + numpy statistical analysis only.
No PyTorch autoencoder (saves ~800MB RAM).
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CVResult:
    anomaly_score: float = 0.0
    method: str = "none"
    reconstruction_error: float | None = None
    edge_density: float | None = None
    variance: float | None = None
    defects_detected: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_score": self.anomaly_score,
            "method": self.method,
            "reconstruction_error": self.reconstruction_error,
            "edge_density": self.edge_density,
            "variance": self.variance,
            "defects_detected": self.defects_detected,
            "details": self.details,
        }


def _numpy_analyse(img_array: np.ndarray) -> CVResult:
    """Pure numpy statistical anomaly detection — no heavy deps."""
    result = CVResult(method="statistical")
    flat = img_array.flatten().astype(np.float32)

    # Variance (low = uniform/suspicious blank image)
    var = float(np.var(flat))
    result.variance = round(var, 2)

    # Mean brightness
    mean_val = float(np.mean(flat))

    # Histogram entropy — measures randomness/noise
    hist, _ = np.histogram(flat, bins=64, range=(0, 255))
    hist_norm = hist / (hist.sum() + 1e-9)
    entropy = float(-np.sum(hist_norm * np.log(hist_norm + 1e-9)))
    max_entropy = float(np.log(64))
    entropy_ratio = entropy / max_entropy

    # Simple edge detection via gradient
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array.astype(np.float32)

    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    edge_density = float((gx + gy) / 255.0)
    result.edge_density = round(edge_density, 4)

    # Anomaly scoring
    var_score = max(0.0, 1.0 - var / 8000.0)        # low variance → suspicious
    noise_score = max(0.0, 1.0 - entropy_ratio)      # low entropy → suspicious
    brightness_score = abs(mean_val - 127.5) / 127.5  # extreme brightness

    result.anomaly_score = round(
        0.4 * var_score + 0.3 * noise_score + 0.3 * brightness_score, 4
    )
    result.defects_detected = result.anomaly_score > 0.5
    result.details = {
        "variance": var,
        "entropy_ratio": round(entropy_ratio, 4),
        "edge_density": edge_density,
        "mean_brightness": round(mean_val, 2),
    }
    return result


def _opencv_analyse(img_array: np.ndarray) -> CVResult:
    """OpenCV analysis with edge + blob detection."""
    try:
        import cv2
        result = CVResult(method="opencv")
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        var = float(np.var(gray))
        result.variance = round(var, 2)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / edges.size
        result.edge_density = round(edge_density, 4)

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_norm = hist / hist.sum()
        import math
        entropy = -float(np.sum(hist_norm * np.log(hist_norm + 1e-9)))
        entropy_ratio = entropy / math.log(256)

        var_score = max(0.0, 1.0 - var / 5000.0)
        edge_score = min(edge_density * 10, 1.0)
        entropy_score = max(0.0, 1.0 - entropy_ratio)

        result.anomaly_score = round(0.4 * var_score + 0.3 * edge_score + 0.3 * entropy_score, 4)
        result.defects_detected = result.anomaly_score > 0.5
        result.details = {"variance": var, "edge_density": edge_density, "entropy_ratio": round(entropy_ratio, 4)}
        return result
    except Exception:
        return _numpy_analyse(img_array)


def analyse_image(image_bytes: bytes) -> CVResult:
    try:
        from PIL import Image
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(pil_img)
    except Exception as e:
        result = CVResult(method="error")
        result.details = {"error": str(e)}
        result.anomaly_score = 0.5
        return result
    return _opencv_analyse(img_array)


def analyse_image_b64(b64_string: str) -> CVResult:
    return analyse_image(base64.b64decode(b64_string))
