"""
NLP Agent - Lightweight version for Render free tier (512MB RAM)
Uses ONLY rule-based processing. No transformers, no spaCy model loading.
Fully functional sentiment + keyword + risk detection without heavy deps.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

RISK_KEYWORDS: set[str] = {
    "fraud", "breach", "hack", "attack", "malware", "ransomware", "phishing",
    "vulnerability", "exploit", "injection", "overflow", "leak", "intrusion",
    "unauthorized", "suspicious", "anomaly", "threat", "danger", "critical",
    "failure", "error", "crash", "corrupt", "illegal", "stolen", "exposed",
    "backdoor", "botnet", "ddos", "spyware", "keylogger", "rootkit", "trojan",
    "virus", "worm", "encrypted", "decrypt", "payload", "command", "control",
    "exfiltration", "privilege", "escalation", "lateral", "movement",
}

NEGATIVE_WORDS: set[str] = {
    "bad", "terrible", "horrible", "awful", "dangerous", "unsafe", "broken",
    "failed", "invalid", "rejected", "denied", "blocked", "compromised",
    "infected", "damaged", "destroyed", "lost", "missing", "stolen", "hacked",
}

POSITIVE_WORDS: set[str] = {
    "good", "great", "excellent", "safe", "secure", "protected", "clean",
    "fixed", "resolved", "patched", "updated", "healthy", "normal", "success",
}


@dataclass
class NLPResult:
    sentiment_label: str = "NEUTRAL"
    sentiment_score: float = 0.5
    keywords: list[str] = field(default_factory=list)
    entities: list[dict[str, str]] = field(default_factory=list)
    risk_keywords_found: list[str] = field(default_factory=list)
    risk_keyword_count: int = 0
    risk_keyword_density: float = 0.0
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


def analyse_text(text: str) -> NLPResult:
    """
    Lightweight NLP using pure Python — no heavy ML libraries.
    Runs in under 10ms, uses <5MB RAM. Perfect for free-tier deployment.
    """
    result = NLPResult()
    tokens = re.findall(r"\b\w+\b", text.lower())
    total = max(len(tokens), 1)

    # --- Sentiment (lexicon-based) ---
    neg_count = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    pos_count = sum(1 for t in tokens if t in POSITIVE_WORDS)
    risk_count = sum(1 for t in tokens if t in RISK_KEYWORDS)

    total_signal = neg_count + pos_count + risk_count
    if total_signal == 0:
        result.sentiment_label = "NEUTRAL"
        result.sentiment_score = 0.5
    elif neg_count + risk_count > pos_count:
        score = min(0.5 + 0.1 * (neg_count + risk_count), 0.99)
        result.sentiment_label = "NEGATIVE"
        result.sentiment_score = round(score, 4)
    else:
        score = min(0.5 + 0.1 * pos_count, 0.99)
        result.sentiment_label = "POSITIVE"
        result.sentiment_score = round(score, 4)

    # --- Keywords (top frequent non-stopwords) ---
    stopwords = {"the","a","an","is","in","on","at","to","for","of","and","or",
                 "but","with","this","that","it","was","are","be","has","have",
                 "i","you","he","she","we","they","my","your","our","their"}
    freq: dict[str, int] = {}
    for t in tokens:
        if t not in stopwords and len(t) > 3:
            freq[t] = freq.get(t, 0) + 1
    result.keywords = [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:10]]

    # --- Entities (capitalised words as pseudo-entities) ---
    result.entities = [
        {"text": m, "label": "ENTITY"}
        for m in re.findall(r"\b[A-Z][A-Za-z]{2,}\b", text)
    ][:10]

    # --- Risk keyword detection ---
    found = list({t for t in tokens if t in RISK_KEYWORDS})
    result.risk_keywords_found = found
    result.risk_keyword_count = sum(1 for t in tokens if t in RISK_KEYWORDS)
    result.risk_keyword_density = round(result.risk_keyword_count / total, 4)

    # --- Composite risk score ---
    sentiment_risk = (
        result.sentiment_score if result.sentiment_label == "NEGATIVE"
        else (1 - result.sentiment_score) * 0.2
    )
    keyword_risk = min(result.risk_keyword_density * 15, 1.0)
    result.nlp_risk_score = round(0.6 * sentiment_risk + 0.4 * keyword_risk, 4)

    return result
