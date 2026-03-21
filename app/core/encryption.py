"""
Custom Encryption Engine
Pipeline: XOR → Byte Shift → Scramble → Base64
All steps are reversible. Does NOT rely solely on AES/Fernet.
"""
import base64
import hashlib
import struct


class CustomCipher:
    """
    Multi-stage symmetric cipher:
      Encrypt: XOR → shift → scramble → base64
      Decrypt: base64 → unscramble → unshift → XOR
    """

    MAGIC = b"\xCA\xFE"          # 2-byte header to validate ciphertext
    VERSION = b"\x01"            # 1-byte version
    SHIFT_AMOUNT = 13            # Caesar-style byte shift

    def __init__(self, secret_key: str):
        if not secret_key:
            raise ValueError("Secret key must not be empty")
        # Derive a 32-byte key via SHA-256 so any string works
        self._key_bytes = hashlib.sha256(secret_key.encode()).digest()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _xor(self, data: bytes) -> bytes:
        """XOR every byte with the repeating key."""
        key = self._key_bytes
        klen = len(key)
        return bytes(b ^ key[i % klen] for i, b in enumerate(data))

    @staticmethod
    def _shift(data: bytes, amount: int) -> bytes:
        """Byte-level Caesar shift (mod 256)."""
        return bytes((b + amount) % 256 for b in data)

    @staticmethod
    def _unshift(data: bytes, amount: int) -> bytes:
        return bytes((b - amount) % 256 for b in data)

    @staticmethod
    def _scramble(data: bytes) -> bytes:
        """
        Block-based scramble: split into 4-byte chunks, reverse each chunk,
        then reverse the chunk order.  100 % reversible with _unscramble.
        """
        block = 4
        chunks = [data[i:i + block] for i in range(0, len(data), block)]
        scrambled = [c[::-1] for c in reversed(chunks)]
        return b"".join(scrambled)

    @staticmethod
    def _unscramble(data: bytes) -> bytes:
        """Inverse of _scramble."""
        block = 4
        chunks = [data[i:i + block] for i in range(0, len(data), block)]
        unscrambled = [c[::-1] for c in reversed(chunks)]
        return b"".join(unscrambled)

    def _add_header(self, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))   # 4-byte big-endian length
        return self.MAGIC + self.VERSION + length + data

    def _strip_header(self, data: bytes) -> bytes:
        if not data.startswith(self.MAGIC + self.VERSION):
            raise ValueError("Invalid ciphertext: bad magic/version header")
        payload_len = struct.unpack(">I", data[3:7])[0]
        payload = data[7:7 + payload_len]
        if len(payload) != payload_len:
            raise ValueError("Ciphertext truncated or corrupted")
        return payload

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt a UTF-8 string.
        Returns a URL-safe Base64 string safe for JSON/HTTP transport.
        """
        raw = plaintext.encode("utf-8")
        stage1 = self._xor(raw)
        stage2 = self._shift(stage1, self.SHIFT_AMOUNT)
        stage3 = self._scramble(stage2)
        framed = self._add_header(stage3)
        return base64.urlsafe_b64encode(framed).decode("ascii")

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt a string produced by encrypt()."""
        framed = base64.urlsafe_b64decode(ciphertext.encode("ascii"))
        stage3 = self._strip_header(framed)
        stage2 = self._unscramble(stage3)
        stage1 = self._unshift(stage2, self.SHIFT_AMOUNT)
        raw = self._xor(stage1)
        return raw.decode("utf-8")

    def encrypt_bytes(self, data: bytes) -> str:
        """Encrypt arbitrary bytes (e.g. image data). Returns Base64 string."""
        stage1 = self._xor(data)
        stage2 = self._shift(stage1, self.SHIFT_AMOUNT)
        stage3 = self._scramble(stage2)
        framed = self._add_header(stage3)
        return base64.urlsafe_b64encode(framed).decode("ascii")

    def decrypt_bytes(self, ciphertext: str) -> bytes:
        """Decrypt bytes produced by encrypt_bytes()."""
        framed = base64.urlsafe_b64decode(ciphertext.encode("ascii"))
        stage3 = self._strip_header(framed)
        stage2 = self._unscramble(stage3)
        stage1 = self._unshift(stage2, self.SHIFT_AMOUNT)
        return self._xor(stage1)


# ------------------------------------------------------------------ #
#  Module-level helper using the app secret                           #
# ------------------------------------------------------------------ #
import os

_cipher: CustomCipher | None = None


def get_cipher() -> CustomCipher:
    global _cipher
    if _cipher is None:
        key = os.getenv("EMIS_SECRET_KEY", "emis-default-dev-key-change-me")
        _cipher = CustomCipher(key)
    return _cipher
