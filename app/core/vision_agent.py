"""
Vision Agent
------------
Performs anomaly / defect detection on an input image.

Two-tier approach:
  1. Primary  – PyTorch autoencoder (reconstruction error as anomaly score)
  2. Fallback – OpenCV statistical analysis (variance, edge density, colour hist)

Returns a CVResult with numeric anomaly_score (0..1) for the risk model.
"""
from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CVResult:
    anomaly_score: float = 0.0        # 0 = normal, 1 = highly anomalous
    method: str = "unknown"
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


# ---------------------------------------------------------------------------
# PyTorch autoencoder (advanced path)
# ---------------------------------------------------------------------------

class ConvAutoencoder:
    """Lightweight Conv autoencoder – trained at inference time on the image
    itself for one-shot anomaly detection (no external dataset needed)."""

    IMG_SIZE = 64
    EPOCHS = 30
    LR = 1e-3

    def __init__(self):
        import torch
        import torch.nn as nn
        self.torch = torch
        self.nn = nn

        class AE(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = nn.Sequential(
                    nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
                    nn.Conv2d(16, 8, 3, stride=2, padding=1), nn.ReLU(),
                )
                self.dec = nn.Sequential(
                    nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
                    nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
                )

            def forward(self, x):
                return self.dec(self.enc(x))

        self.model = AE()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR)
        self.criterion = nn.MSELoss()

    def _preprocess(self, img_array: np.ndarray):
        import cv2
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
        resized = cv2.resize(gray, (self.IMG_SIZE, self.IMG_SIZE))
        tensor = self.torch.tensor(resized, dtype=self.torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        return tensor

    def fit_and_score(self, img_array: np.ndarray) -> float:
        """Train on image, return reconstruction error as anomaly score."""
        x = self._preprocess(img_array)
        self.model.train()
        for _ in range(self.EPOCHS):
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, x)
            loss.backward()
            self.optimizer.step()

        self.model.eval()
        with self.torch.no_grad():
            out = self.model(x)
            mse = self.criterion(out, x).item()

        # Normalise: typical well-reconstructed images score < 0.01
        score = min(mse * 50, 1.0)
        return round(score, 4)


# ---------------------------------------------------------------------------
# OpenCV fallback
# ---------------------------------------------------------------------------

def _opencv_analyse(img_array: np.ndarray) -> CVResult:
    try:
        import cv2
    except ImportError:
        return _numpy_only_analyse(img_array)

    result = CVResult(method="opencv")

    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array

    # Variance (low = uniform / suspicious)
    var = float(np.var(gray))
    result.variance = round(var, 2)

    # Edge density via Canny
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0)) / edges.size
    result.edge_density = round(edge_density, 4)

    # Colour histogram anomaly (check for unusual distributions)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / hist.sum()
    entropy = -float(np.sum(hist_norm * np.log(hist_norm + 1e-9)))
    max_entropy = math.log(256)
    entropy_ratio = entropy / max_entropy

    # Blob detection for defects
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 200
    params.filterByArea = True
    params.minArea = 50
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)
    result.defects_detected = len(keypoints) > 3

    # Composite anomaly score
    var_score = max(0.0, 1.0 - var / 5000.0)          # low variance → higher risk
    edge_score = min(edge_density * 10, 1.0)
    entropy_score = max(0.0, 1.0 - entropy_ratio)
    defect_score = 0.3 if result.defects_detected else 0.0

    result.anomaly_score = round(
        0.3 * var_score + 0.3 * edge_score + 0.2 * entropy_score + 0.2 * defect_score,
        4
    )
    result.details = {
        "variance": var,
        "edge_density": edge_density,
        "entropy_ratio": round(entropy_ratio, 4),
        "blob_count": len(keypoints),
    }
    return result


def _numpy_only_analyse(img_array: np.ndarray) -> CVResult:
    """Pure-numpy fallback when OpenCV is unavailable."""
    result = CVResult(method="numpy_fallback")
    flat = img_array.flatten().astype(np.float32)
    var = float(np.var(flat))
    result.variance = round(var, 2)
    result.anomaly_score = round(max(0.0, 1.0 - var / 10000.0), 4)
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyse_image(image_bytes: bytes) -> CVResult:
    """
    Accepts raw image bytes.
    Tries PyTorch autoencoder first, falls back to OpenCV, then numpy.
    """
    try:
        from PIL import Image
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(pil_img)
    except Exception as e:
        result = CVResult(method="error")
        result.details = {"error": str(e)}
        result.anomaly_score = 0.5
        return result

    # Try PyTorch autoencoder
    try:
        ae = ConvAutoencoder()
        recon_err = ae.fit_and_score(img_array)
        result = CVResult(
            anomaly_score=recon_err,
            method="pytorch_autoencoder",
            reconstruction_error=recon_err,
            defects_detected=recon_err > 0.5,
        )
        return result
    except Exception:
        pass

    # Fall back to OpenCV
    return _opencv_analyse(img_array)


def analyse_image_b64(b64_string: str) -> CVResult:
    """Accepts base64-encoded image string."""
    image_bytes = base64.b64decode(b64_string)
    return analyse_image(image_bytes)
