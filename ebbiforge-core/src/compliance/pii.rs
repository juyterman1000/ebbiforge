//! PII Detection and Redaction
//!
//! Automatically detect and mask sensitive data:
//! - Email addresses
//! - Phone numbers
//! - Social Security Numbers
//! - Credit card numbers
//! - API keys

use pyo3::prelude::*;
use regex::Regex;
use std::sync::LazyLock;

/// PII pattern definitions
static EMAIL_PATTERN: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").ok());

static PHONE_PATTERN: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b").ok());

static SSN_PATTERN: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").ok());

static CREDIT_CARD_PATTERN: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b").ok());

static API_KEY_PATTERN: LazyLock<Option<Regex>> = LazyLock::new(|| {
    // Matches sk-xxx style keys and api_key=xxx style
    Regex::new(r"(sk-[a-zA-Z0-9]{20,}|api[_-]?key[=:]\s*[a-zA-Z0-9]{16,})").ok()
});

static ADDRESS_PATTERN: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(r"(?i)\b\d{1,5}\s(?:[A-Za-z0-9#\s]+)\s(?:Street|St|Avenue|Ave|Road|Rd|Highway|Hwy|Square|Sq|Trail|Trl|Drive|Dr|Court|Ct|Parkway|Pkwy|Circle|Cir|Boulevard|Blvd)\b").ok()
});

static DOB_PATTERN: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b").ok()
});

static PASSPORT_PATTERN: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(r"\b[A-Z0-9]{6,9}\b").ok() // Generalized passport-like strings
});

static BIOMETRIC_PATTERN: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(fingerprint|retina\sscan|facial\srecognition|biometric\sdata|voice\ssignature)\b",
    )
    .ok()
});

static DRIVERS_LICENSE_PATTERN: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"\b[A-Z0-9]{1,3}\d{6,14}\b").ok());

static BANK_ACCOUNT_PATTERN: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"\b\d{8,17}\b").ok());

static MEDICAL_PATTERN: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(prescription|medical\srecord|health\shistory|diagnosis|treatment)\b").ok()
});

static DIGITAL_ID_PATTERN: LazyLock<Option<Regex>> = LazyLock::new(|| {
    // IP address, Cookie ID pattern
    Regex::new(r"\b(?:\d{1,3}\.){3}\d{1,3}\b|(?i)cookie[_-]?id[=:]\s*[a-zA-Z0-9]{16,}").ok()
});

static DEMOGRAPHIC_PATTERN: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"(?i)\b(race|religion|gender|sexual\sorientation)\b").ok());

static GPS_PATTERN: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"\b-?\d{1,3}\.\d{4,},\s*-?\d{1,3}\.\d{4,}\b").ok());

/// PII detection result
#[derive(Debug, Clone)]
#[pyclass]
pub struct PIIMatch {
    #[pyo3(get)]
    pub pii_type: String,
    #[pyo3(get)]
    pub value: String,
    #[pyo3(get)]
    pub start: usize,
    #[pyo3(get)]
    pub end: usize,
}

#[pymethods]
impl PIIMatch {
    pub fn __repr__(&self) -> String {
        format!(
            "PIIMatch({}: '{}' at {}..{})",
            self.pii_type, self.value, self.start, self.end
        )
    }
}

/// PII redactor
#[pyclass]
pub struct PIIRedactor {
    redaction_char: char,
}

#[pymethods]
impl PIIRedactor {
    #[new]
    #[pyo3(signature = (redaction_char = '*'))]
    pub fn new(redaction_char: char) -> Self {
        PIIRedactor { redaction_char }
    }

    /// Detect all PII in text
    pub fn detect_pii(&self, text: &str) -> Vec<PIIMatch> {
        let mut matches = Vec::new();

        // Email
        if let Some(pattern) = EMAIL_PATTERN.as_ref() {
            for m in pattern.find_iter(text) {
                matches.push(PIIMatch {
                    pii_type: "Email".to_string(),
                    value: m.as_str().to_string(),
                    start: m.start(),
                    end: m.end(),
                });
            }
        }

        // Phone
        if let Some(pattern) = PHONE_PATTERN.as_ref() {
            for m in pattern.find_iter(text) {
                matches.push(PIIMatch {
                    pii_type: "Phone".to_string(),
                    value: m.as_str().to_string(),
                    start: m.start(),
                    end: m.end(),
                });
            }
        }

        // SSN
        if let Some(pattern) = SSN_PATTERN.as_ref() {
            for m in pattern.find_iter(text) {
                matches.push(PIIMatch {
                    pii_type: "SSN".to_string(),
                    value: m.as_str().to_string(),
                    start: m.start(),
                    end: m.end(),
                });
            }
        }

        // Credit Card
        if let Some(pattern) = CREDIT_CARD_PATTERN.as_ref() {
            for m in pattern.find_iter(text) {
                matches.push(PIIMatch {
                    pii_type: "CreditCard".to_string(),
                    value: m.as_str().to_string(),
                    start: m.start(),
                    end: m.end(),
                });
            }
        }

        // API Key
        if let Some(pattern) = API_KEY_PATTERN.as_ref() {
            for m in pattern.find_iter(text) {
                matches.push(PIIMatch {
                    pii_type: "APIKey".to_string(),
                    value: m.as_str().to_string(),
                    start: m.start(),
                    end: m.end(),
                });
            }
        }

        // Address
        if let Some(pattern) = ADDRESS_PATTERN.as_ref() {
            for m in pattern.find_iter(text) {
                matches.push(PIIMatch {
                    pii_type: "Address".to_string(),
                    value: m.as_str().to_string(),
                    start: m.start(),
                    end: m.end(),
                });
            }
        }

        // DOB
        if let Some(pattern) = DOB_PATTERN.as_ref() {
            for m in pattern.find_iter(text) {
                matches.push(PIIMatch {
                    pii_type: "DOB".to_string(),
                    value: m.as_str().to_string(),
                    start: m.start(),
                    end: m.end(),
                });
            }
        }

        // Passport
        if let Some(pattern) = PASSPORT_PATTERN.as_ref() {
            for m in pattern.find_iter(text) {
                matches.push(PIIMatch {
                    pii_type: "Passport".to_string(),
                    value: m.as_str().to_string(),
                    start: m.start(),
                    end: m.end(),
                });
            }
        }

        // Biometric
        if let Some(pattern) = BIOMETRIC_PATTERN.as_ref() {
            for m in pattern.find_iter(text) {
                matches.push(PIIMatch {
                    pii_type: "Biometric".to_string(),
                    value: m.as_str().to_string(),
                    start: m.start(),
                    end: m.end(),
                });
            }
        }

        // Drivers License
        if let Some(pattern) = DRIVERS_LICENSE_PATTERN.as_ref() {
            for m in pattern.find_iter(text) {
                matches.push(PIIMatch {
                    pii_type: "DriversLicense".to_string(),
                    value: m.as_str().to_string(),
                    start: m.start(),
                    end: m.end(),
                });
            }
        }

        // Bank Account
        if let Some(pattern) = BANK_ACCOUNT_PATTERN.as_ref() {
            for m in pattern.find_iter(text) {
                matches.push(PIIMatch {
                    pii_type: "BankAccount".to_string(),
                    value: m.as_str().to_string(),
                    start: m.start(),
                    end: m.end(),
                });
            }
        }

        // Medical
        if let Some(pattern) = MEDICAL_PATTERN.as_ref() {
            for m in pattern.find_iter(text) {
                matches.push(PIIMatch {
                    pii_type: "Medical".to_string(),
                    value: m.as_str().to_string(),
                    start: m.start(),
                    end: m.end(),
                });
            }
        }

        // Digital ID
        if let Some(pattern) = DIGITAL_ID_PATTERN.as_ref() {
            for m in pattern.find_iter(text) {
                matches.push(PIIMatch {
                    pii_type: "DigitalID".to_string(),
                    value: m.as_str().to_string(),
                    start: m.start(),
                    end: m.end(),
                });
            }
        }

        // Demographic
        if let Some(pattern) = DEMOGRAPHIC_PATTERN.as_ref() {
            for m in pattern.find_iter(text) {
                matches.push(PIIMatch {
                    pii_type: "Demographic".to_string(),
                    value: m.as_str().to_string(),
                    start: m.start(),
                    end: m.end(),
                });
            }
        }

        // GPS
        if let Some(pattern) = GPS_PATTERN.as_ref() {
            for m in pattern.find_iter(text) {
                matches.push(PIIMatch {
                    pii_type: "GPS".to_string(),
                    value: m.as_str().to_string(),
                    start: m.start(),
                    end: m.end(),
                });
            }
        }

        matches
    }

    /// Redact all PII from text
    pub fn redact(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Redact in reverse order to preserve positions
        let mut matches = self.detect_pii(text);
        matches.sort_by(|a, b| b.start.cmp(&a.start));

        for m in matches {
            let redacted = self.redaction_char.to_string().repeat(m.value.len());
            result.replace_range(m.start..m.end, &redacted);
        }

        result
    }

    /// Check if text is clean (no PII)
    pub fn is_clean(&self, text: &str) -> bool {
        self.detect_pii(text).is_empty()
    }
}

impl Default for PIIRedactor {
    fn default() -> Self {
        Self::new('*')
    }
}
