"""
NLP Agent
---------
Performs:
  1. Sentiment analysis  (transformers pipeline)
  2. Named-entity / keyword extraction  (spaCy)
  3. Risk keyword detection  (rule-based vocabulary)

Returns a structured NLPResult with numeric features ready for the risk model.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Risk vocabulary – extend as needed
# ---------------------------------------------------------------------------
RISK_KEYWORDS: set[str] = {
    "fraud", "breach", "hack", "attack", "malware", "ransomware", "phishing",
    "vulnerability", "exploit", "injection", "overflow", "leak", "intrusion",
    "unauthorized", "suspicious", "anomaly", "threat", "danger", "critical",
    "failure", "error", "crash", "corrupt", "illegal", "stolen", "exposed",
    "backdoor", "botnet", "ddos", "spyware", "keylogger", "rootkit", "trojan",
}

NEGATIVE_SENTIMENT_WORDS: set[str] = {
    "bad", "terrible", "horrible", "awful", "dangerous", "unsafe", "broken",
    "failed", "invalid", "rejected", "denied", "blocked",
}


@dataclass
class NLPResult:
    # Sentiment
    sentiment_label: str = "NEUTRAL"      # POSITIVE / NEGATIVE / NEUTRAL
    sentiment_score: float = 0.5          # 0..1 confidence

    # Keywords / entities
    keywords: list[str] = field(default_factory=list)
    entities: list[dict[str, str]] = field(default_factory=list)

    # Risk
    risk_keywords_found: list[str] = field(default_factory=list)
    risk_keyword_count: int = 0
    risk_keyword_density: float = 0.0    # risk keywords / total tokens

    # Derived numeric feature for risk model (0..1)
    nlp_risk_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentiment_label": self.sentiment_label,
            "sentiment_score": self.sentiment_score,
            "keywords": self.keywords,
            "entities": self.entities,
            "risk_keywords_found": self.risk_keywords_found,
            "risk_keyword_count": self.risk_keyword_count,
            "risk_keyword_density": self.risk_keyword_density,
            "nlp_risk_score": self.nlp_risk_score,
        }


# ---------------------------------------------------------------------------
# Lazy model loading – only imports heavy libs when first called
# ---------------------------------------------------------------------------
_sentiment_pipeline = None
_nlp_model = None


def _load_sentiment():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            _sentiment_pipeline = hf_pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True,
                max_length=512,
            )
        except Exception:
            _sentiment_pipeline = "fallback"
    return _sentiment_pipeline


def _load_spacy():
    global _nlp_model
    if _nlp_model is None:
        try:
            import spacy
            try:
                _nlp_model = spacy.load("en_core_web_sm")
            except OSError:
                from spacy.cli import download as spacy_download
                spacy_download("en_core_web_sm")
                _nlp_model = spacy.load("en_core_web_sm")
        except Exception:
            _nlp_model = "fallback"
    return _nlp_model


# ---------------------------------------------------------------------------
# Fallback implementations (no heavy deps required)
# ---------------------------------------------------------------------------

def _fallback_sentiment(text: str) -> tuple[str, float]:
    """Simple lexicon-based sentiment when transformers unavailable."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    neg = sum(1 for t in tokens if t in NEGATIVE_SENTIMENT_WORDS)
    risk = sum(1 for t in tokens if t in RISK_KEYWORDS)
    if neg + risk > 1:
        score = min(0.5 + 0.1 * (neg + risk), 0.99)
        return "NEGATIVE", score
    return "POSITIVE", 0.6


def _fallback_entities(text: str) -> tuple[list[str], list[dict]]:
    """Regex-based keyword + entity extraction fallback."""
    tokens = re.findall(r"\b[A-Za-z]{4,}\b", text)
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t.lower()] = freq.get(t.lower(), 0) + 1
    keywords = [w for w, c in sorted(freq.items(), key=lambda x: -x[1])[:10]]
    # Capitalised words as pseudo-entities
    entities = [
        {"text": m, "label": "ENTITY"}
        for m in re.findall(r"\b[A-Z][a-z]{2,}\b", text)
    ][:10]
    return keywords, entities


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyse_text(text: str) -> NLPResult:
    """Run the full NLP pipeline and return an NLPResult."""
    result = NLPResult()
    tokens = re.findall(r"\b\w+\b", text.lower())
    total_tokens = max(len(tokens), 1)

    # ---- 1. Sentiment -------------------------------------------------- #
    pipeline = _load_sentiment()
    if pipeline != "fallback":
        try:
            out = pipeline(text[:512])[0]
            raw_label = out["label"].upper()
            result.sentiment_label = raw_label if raw_label in ("POSITIVE", "NEGATIVE") else "NEUTRAL"
            result.sentiment_score = float(out["score"])
        except Exception:
            result.sentiment_label, result.sentiment_score = _fallback_sentiment(text)
    else:
        result.sentiment_label, result.sentiment_score = _fallback_sentiment(text)

    # ---- 2. Keywords & entities ---------------------------------------- #
    nlp = _load_spacy()
    if nlp != "fallback":
        try:
            doc = nlp(text[:100_000])
            result.keywords = list({
                chunk.text.lower() for chunk in doc.noun_chunks
                if len(chunk.text) > 3
            })[:15]
            result.entities = [
                {"text": ent.text, "label": ent.label_}
                for ent in doc.ents
            ][:20]
        except Exception:
            result.keywords, result.entities = _fallback_entities(text)
    else:
        result.keywords, result.entities = _fallback_entities(text)

    # ---- 3. Risk keyword detection ------------------------------------- #
    found = [t for t in tokens if t in RISK_KEYWORDS]
    result.risk_keywords_found = list(set(found))
    result.risk_keyword_count = len(found)
    result.risk_keyword_density = len(found) / total_tokens

    # ---- 4. Composite NLP risk score ----------------------------------- #
    sentiment_risk = (
        result.sentiment_score if result.sentiment_label == "NEGATIVE" else
        (1 - result.sentiment_score) * 0.3
    )
    keyword_risk = min(result.risk_keyword_density * 20, 1.0)
    result.nlp_risk_score = round(0.6 * sentiment_risk + 0.4 * keyword_risk, 4)

    return result
