import re
import json
import base64
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class PrivacyFilter:
    """Exhaustive PII Filter based on User-defined categories"""
    
    PATTERNS = {
        # Direct PII
        "GovernmentID": r"\b\d{3}-\d{2}-\d{4}\b|\b(?=.*\d)[A-Z0-9]{6,14}\b", # SSN, Passport (Must have digit)
        "Financial": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b|\b\d{8,17}\b", # CC, Bank Account
        "Medical": r"\b(prescription|medical\srecord|health\shistory|diagnosis|treatment)\b",
        "Biometric": r"\b(fingerprint|retina\sscan|facial\srecognition|biometric\sdata|voice\ssignature)\b",
        
        # Indirect PII
        "Contact": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", # Email, Phone
        "Address": r"\b\d{1,5}\s(?:[A-Za-z0-9#\s]+)\s(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Boulevard|Blvd)\b",
        "DigitalID": r"\b(?:\d{1,3}\.){3}\d{1,3}\b|cookie[_-]?id[=:]\s*[a-zA-Z0-9]{16,}", # IP, Cookie
        "Demographic": r"\b(race|religion|gender|sexual\sorientation|date\sof\sbirth)\b",
        "Location": r"\b-?\d{1,3}\.\d{4,},\s*-?\d{1,3}\.\d{4,}\b", # GPS
    }

    @classmethod
    def redact(cls, text: str, mask_char="*") -> str:
        redacted = text
        for p_type, pattern in cls.PATTERNS.items():
            matches = re.finditer(pattern, redacted, re.IGNORECASE)
            for m in matches:
                val = m.group()
                redacted = redacted.replace(val, mask_char * len(val))
        return redacted

class Vault:
    """AES-256 Encryption at Rest"""
    
    def __init__(self, key_hex: str = None):
        if not key_hex:
            self.key = AESGCM.generate_key(bit_length=256)
        else:
            self.key = bytes.fromhex(key_hex)
        self.aesgcm = AESGCM(self.key)

    def encrypt(self, data: str) -> str:
        nonce = os.urandom(12)
        ct = self.aesgcm.encrypt(nonce, data.encode(), None)
        return base64.b64encode(nonce + ct).decode('utf-8')

    def decrypt(self, b64_data: str) -> str:
        data = base64.b64decode(b64_data)
        nonce = data[:12]
        ct = data[12:]
        return self.aesgcm.decrypt(nonce, ct, None).decode('utf-8')
