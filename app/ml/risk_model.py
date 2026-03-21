"""
Risk Scoring Model
------------------
Takes NLPResult + CVResult features → unified risk score (0..1).

Uses XGBoost if available, else Random Forest, else Logistic Regression.
Model is trained on synthetic rule-based data at startup (no external dataset).
In production, replace with a pre-trained model loaded from disk.
"""
from __future__ import annotations

import os
import pickle
import numpy as np
from dataclasses import dataclass
from typing import Any

from app.core.nlp_agent import NLPResult
from app.core.vision_agent import CVResult


@dataclass
class RiskResult:
    risk_score: float           # 0..1
    risk_level: str             # LOW / MEDIUM / HIGH / CRITICAL
    model_used: str
    feature_importance: dict[str, float]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "model_used": self.model_used,
            "feature_importance": self.feature_importance,
            "explanation": self.explanation,
        }


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "nlp_risk_score",
    "sentiment_negative",
    "risk_keyword_count",
    "risk_keyword_density",
    "entity_count",
    "keyword_count",
    "cv_anomaly_score",
    "cv_defect_flag",
    "cv_edge_density",
    "cv_variance_norm",
    "combined_risk",       # nlp * cv
]


def extract_features(nlp: NLPResult, cv: CVResult) -> np.ndarray:
    sentiment_neg = 1.0 if nlp.sentiment_label == "NEGATIVE" else 0.0
    cv_edge = cv.edge_density if cv.edge_density is not None else 0.0
    cv_var = min((cv.variance or 0) / 10000.0, 1.0)
    combined = nlp.nlp_risk_score * cv.anomaly_score

    features = [
        nlp.nlp_risk_score,
        sentiment_neg,
        min(nlp.risk_keyword_count / 10.0, 1.0),
        min(nlp.risk_keyword_density * 20, 1.0),
        min(len(nlp.entities) / 20.0, 1.0),
        min(len(nlp.keywords) / 15.0, 1.0),
        cv.anomaly_score,
        1.0 if cv.defects_detected else 0.0,
        min(cv_edge, 1.0),
        cv_var,
        combined,
    ]
    return np.array(features, dtype=np.float32).reshape(1, -1)


# ---------------------------------------------------------------------------
# Synthetic training data generator
# ---------------------------------------------------------------------------

def _generate_training_data(n: int = 2000):
    rng = np.random.default_rng(42)
    X, y = [], []

    for _ in range(n):
        nlp_risk = rng.uniform(0, 1)
        sent_neg = float(rng.random() < (0.3 + 0.5 * nlp_risk))
        rk_count = rng.uniform(0, 1)
        rk_density = rng.uniform(0, 1) * nlp_risk
        ent_count = rng.uniform(0, 1)
        kw_count = rng.uniform(0, 1)
        cv_anom = rng.uniform(0, 1)
        cv_defect = float(rng.random() < cv_anom * 0.7)
        cv_edge = rng.uniform(0, 1)
        cv_var = rng.uniform(0, 1)
        combined = nlp_risk * cv_anom

        features = [nlp_risk, sent_neg, rk_count, rk_density, ent_count,
                    kw_count, cv_anom, cv_defect, cv_edge, cv_var, combined]

        # Label: weighted combination with noise
        score = (0.35 * nlp_risk + 0.25 * cv_anom + 0.15 * sent_neg +
                 0.15 * rk_density + 0.1 * cv_defect)
        label = int(score > 0.4)   # binary for classifier
        X.append(features)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y)


# ---------------------------------------------------------------------------
# Model loader (singleton)
# ---------------------------------------------------------------------------

_model = None
_model_name = "unknown"
_MODEL_PATH = "/tmp/emis_risk_model.pkl"


def _train_model():
    global _model, _model_name
    X, y = _generate_training_data()

    try:
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0,
        )
        clf.fit(X, y)
        _model = clf
        _model_name = "XGBoost"
    except ImportError:
        try:
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X, y)
            _model = clf
            _model_name = "RandomForest"
        except ImportError:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, random_state=42)),
            ])
            clf.fit(X, y)
            _model = clf
            _model_name = "LogisticRegression"

    with open(_MODEL_PATH, "wb") as f:
        pickle.dump((_model, _model_name), f)


def _get_model():
    global _model, _model_name
    if _model is not None:
        return _model, _model_name
    if os.path.exists(_MODEL_PATH):
        with open(_MODEL_PATH, "rb") as f:
            _model, _model_name = pickle.load(f)
        return _model, _model_name
    _train_model()
    return _model, _model_name


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _risk_level(score: float) -> str:
    if score < 0.25:
        return "LOW"
    if score < 0.50:
        return "MEDIUM"
    if score < 0.75:
        return "HIGH"
    return "CRITICAL"


def _get_importance(model, name: str) -> dict[str, float]:
    try:
        if name == "XGBoost":
            imp = model.feature_importances_
        elif name == "RandomForest":
            imp = model.feature_importances_
        else:
            imp = np.abs(model.named_steps["lr"].coef_[0])
        imp = imp / (imp.sum() + 1e-9)
        return {k: round(float(v), 4) for k, v in zip(FEATURE_NAMES, imp)}
    except Exception:
        return {k: round(1 / len(FEATURE_NAMES), 4) for k in FEATURE_NAMES}


def score_risk(nlp: NLPResult, cv: CVResult) -> RiskResult:
    model, name = _get_model()
    X = extract_features(nlp, cv)

    try:
        proba = model.predict_proba(X)[0]
        risk_score = round(float(proba[1]), 4)
    except Exception:
        # Hard fallback: weighted average of raw features
        risk_score = round(0.5 * nlp.nlp_risk_score + 0.5 * cv.anomaly_score, 4)

    level = _risk_level(risk_score)
    importance = _get_importance(model, name)

    top_feature = max(importance, key=importance.get)
    explanation = (
        f"Risk level is {level} (score={risk_score:.2f}). "
        f"Primary driver: {top_feature.replace('_', ' ')}. "
        f"NLP risk={nlp.nlp_risk_score:.2f}, CV anomaly={cv.anomaly_score:.2f}. "
        f"Detected {nlp.risk_keyword_count} risk keyword(s): "
        f"{', '.join(nlp.risk_keywords_found[:5]) or 'none'}."
    )

    return RiskResult(
        risk_score=risk_score,
        risk_level=level,
        model_used=name,
        feature_importance=importance,
        explanation=explanation,
    )
