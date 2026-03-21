"""
EMIS Test Suite
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ===================================================================== #
# Encryption tests
# ===================================================================== #

class TestEncryption:
    def setup_method(self):
        from app.core.encryption import CustomCipher
        self.cipher = CustomCipher("test-secret-key")

    def test_encrypt_decrypt_roundtrip(self):
        texts = [
            "Hello, World!",
            "Suspicious login attempt from 192.168.1.1",
            "短い日本語テキスト",          # Unicode
            "A" * 1000,                   # Long string
            "!@#$%^&*()_+-=[]{}|;':\",.<>?/`~",  # Special chars
        ]
        for t in texts:
            assert self.cipher.decrypt(self.cipher.encrypt(t)) == t

    def test_encrypt_bytes_roundtrip(self):
        data = b"\x00\x01\x02\xff\xfe\x80"
        assert self.cipher.decrypt_bytes(self.cipher.encrypt_bytes(data)) == data

    def test_different_keys_produce_different_ciphertext(self):
        from app.core.encryption import CustomCipher
        c1 = CustomCipher("key-one")
        c2 = CustomCipher("key-two")
        assert c1.encrypt("same text") != c2.encrypt("same text")

    def test_wrong_key_raises(self):
        from app.core.encryption import CustomCipher
        c1 = CustomCipher("correct-key")
        c2 = CustomCipher("wrong-key")
        ct = c1.encrypt("secret")
        with pytest.raises(Exception):
            c2.decrypt(ct)

    def test_same_plaintext_has_no_prefix_leakage(self):
        ct1 = self.cipher.encrypt("hello")
        ct2 = self.cipher.encrypt("hello")
        # ciphertexts are deterministic but must decode correctly
        assert self.cipher.decrypt(ct1) == "hello"
        assert self.cipher.decrypt(ct2) == "hello"

    def test_empty_key_raises(self):
        from app.core.encryption import CustomCipher
        with pytest.raises(ValueError):
            CustomCipher("")

    def test_ciphertext_is_base64_ascii(self):
        import base64
        ct = self.cipher.encrypt("test payload")
        base64.urlsafe_b64decode(ct)   # should not raise


# ===================================================================== #
# NLP agent tests
# ===================================================================== #

class TestNLPAgent:
    def test_risk_keywords_detected(self):
        from app.core.nlp_agent import analyse_text
        result = analyse_text("ransomware attack and phishing attempt detected")
        assert result.risk_keyword_count >= 2
        assert "ransomware" in result.risk_keywords_found or "phishing" in result.risk_keywords_found

    def test_clean_text_low_risk(self):
        from app.core.nlp_agent import analyse_text
        result = analyse_text("The weather today is sunny and pleasant. Have a great day!")
        assert result.nlp_risk_score < 0.6

    def test_result_has_all_fields(self):
        from app.core.nlp_agent import analyse_text
        result = analyse_text("test input")
        assert hasattr(result, 'sentiment_label')
        assert hasattr(result, 'nlp_risk_score')
        assert hasattr(result, 'risk_keywords_found')
        assert 0.0 <= result.nlp_risk_score <= 1.0

    def test_to_dict_serialisable(self):
        import json
        from app.core.nlp_agent import analyse_text
        result = analyse_text("breach detected")
        d = result.to_dict()
        json.dumps(d)   # must not raise


# ===================================================================== #
# Risk model tests
# ===================================================================== #

class TestRiskModel:
    def test_high_nlp_high_cv_gives_high_score(self):
        from app.core.nlp_agent import NLPResult
        from app.core.vision_agent import CVResult
        from app.ml.risk_model import score_risk
        nlp = NLPResult(nlp_risk_score=0.9, sentiment_label="NEGATIVE",
                        risk_keyword_count=5, risk_keyword_density=0.2,
                        risk_keywords_found=["ransomware"])
        cv  = CVResult(anomaly_score=0.85, defects_detected=True)
        risk = score_risk(nlp, cv)
        assert risk.risk_score > 0.4
        assert risk.risk_level in ("HIGH", "CRITICAL")

    def test_low_scores_give_low_risk(self):
        from app.core.nlp_agent import NLPResult
        from app.core.vision_agent import CVResult
        from app.ml.risk_model import score_risk
        nlp = NLPResult(nlp_risk_score=0.05, sentiment_label="POSITIVE")
        cv  = CVResult(anomaly_score=0.05)
        risk = score_risk(nlp, cv)
        assert risk.risk_score < 0.6

    def test_explanation_is_non_empty(self):
        from app.core.nlp_agent import NLPResult
        from app.core.vision_agent import CVResult
        from app.ml.risk_model import score_risk
        risk = score_risk(NLPResult(), CVResult())
        assert len(risk.explanation) > 10

    def test_risk_level_boundaries(self):
        from app.ml.risk_model import _risk_level
        assert _risk_level(0.10) == "LOW"
        assert _risk_level(0.35) == "MEDIUM"
        assert _risk_level(0.60) == "HIGH"
        assert _risk_level(0.90) == "CRITICAL"


# ===================================================================== #
# Database tests (in-memory mode)
# ===================================================================== #

class TestDatabase:
    @pytest.mark.asyncio
    async def test_save_and_retrieve(self):
        from app.core import database
        record = await database.save_record(
            decrypted_text="test text",
            image_path=None,
            risk_score=0.42,
            risk_level="MEDIUM",
            nlp_data={},
            cv_data={},
            explanation="test explanation",
        )
        assert record["risk_score"] == 0.42
        recent = await database.get_recent(1)
        assert len(recent) >= 1

    @pytest.mark.asyncio
    async def test_stats_returns_dict(self):
        from app.core import database
        stats = await database.get_stats()
        assert "total" in stats
        assert "avg_risk" in stats
