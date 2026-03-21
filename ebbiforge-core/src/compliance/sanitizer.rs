//! Input Sanitizer for prompt injection defense
//!
//! Detects and blocks common prompt injection patterns.

use pyo3::prelude::*;
use regex::Regex;
use std::sync::LazyLock;
use tracing::{info, warn};

/// Known injection patterns
static INJECTION_PATTERNS: LazyLock<Vec<(Option<Regex>, &'static str)>> = LazyLock::new(|| {
    vec![
        // System prompt overrides
        (
            Regex::new(r"(?i)ignore\s+(previous|all|above)\s+(instructions?|prompts?)").ok(),
            "System prompt override attempt",
        ),
        (
            Regex::new(r"(?i)forget\s+(everything|all|previous)").ok(),
            "Memory manipulation attempt",
        ),
        (
            Regex::new(r"(?i)you\s+are\s+now\s+a").ok(),
            "Role override attempt",
        ),
        (
            Regex::new(r"(?i)new\s+instructions?:").ok(),
            "Instruction injection",
        ),
        (
            Regex::new(r"(?i)disregard\s+(your|the)\s+(rules|guidelines)").ok(),
            "Rule bypass attempt",
        ),
        // Jailbreak patterns
        (
            Regex::new(r"(?i)jailbreak\s*mode").ok(),
            "Jailbreak attempt",
        ),
        (
            Regex::new(r"(?i)developer\s+mode").ok(),
            "Developer mode jailbreak",
        ),
        (
            Regex::new(r"(?i)pretend\s+you\s+(can|have|are)").ok(),
            "Capability bypass attempt",
        ),
        // Data exfiltration
        (
            Regex::new(r"(?i)repeat\s+(back|after|everything)").ok(),
            "Data exfiltration attempt",
        ),
        (
            Regex::new(r"(?i)print\s+(your|the)\s+(system|initial)\s+prompt").ok(),
            "System prompt extraction",
        ),
        // Code injection
        (
            Regex::new(r"(?i)```\s*(python|javascript|bash|sh)\s*\n.*exec\(").ok(),
            "Code execution injection",
        ),
        (
            Regex::new(r"(?i)os\.system\s*\(").ok(),
            "OS command injection",
        ),
        (
            Regex::new(r"(?i)subprocess\.(run|call|Popen)").ok(),
            "Subprocess injection",
        ),
    ]
});

/// Result of input sanitization
#[derive(Clone)]
#[pyclass]
pub struct SanitizeResult {
    #[pyo3(get)]
    pub is_safe: bool,
    #[pyo3(get)]
    pub threats: Vec<String>,
    #[pyo3(get)]
    pub sanitized_input: String,
}

#[pymethods]
impl SanitizeResult {
    pub fn __repr__(&self) -> String {
        if self.is_safe {
            "SanitizeResult(SAFE)".to_string()
        } else {
            format!("SanitizeResult(BLOCKED: {:?})", self.threats)
        }
    }
}

/// Input sanitizer for prompt injection defense
#[pyclass]
pub struct InputSanitizer {
    pub block_on_threat: bool,
}

#[pymethods]
impl InputSanitizer {
    #[new]
    #[pyo3(signature = (block_on_threat = true))]
    pub fn new(block_on_threat: bool) -> Self {
        info!(
            "[Sanitizer] Initialized with {} injection patterns",
            INJECTION_PATTERNS.len()
        );
        InputSanitizer { block_on_threat }
    }

    /// Check input for injection threats
    pub fn check(&self, input: String) -> SanitizeResult {
        let mut threats = Vec::new();

        for (pattern, description) in INJECTION_PATTERNS.iter() {
            if let Some(p) = pattern {
                if p.is_match(&input) {
                    threats.push(description.to_string());
                }
            }
        }

        if threats.is_empty() {
            SanitizeResult {
                is_safe: true,
                threats: vec![],
                sanitized_input: input,
            }
        } else {
            warn!("🚫 [Sanitizer] Blocked: {:?}", threats);
            SanitizeResult {
                is_safe: false,
                threats,
                sanitized_input: if self.block_on_threat {
                    "[BLOCKED: Potential injection detected]".to_string()
                } else {
                    input
                },
            }
        }
    }

    /// Sanitize input by removing dangerous patterns
    pub fn sanitize(&self, input: String) -> String {
        let mut result = input;

        for (pattern, _) in INJECTION_PATTERNS.iter() {
            if let Some(p) = pattern {
                result = p.replace_all(&result, "[REDACTED]").to_string();
            }
        }

        result
    }

    /// Check if input is safe (simple boolean)
    pub fn is_safe(&self, input: String) -> bool {
        for (pattern, _) in INJECTION_PATTERNS.iter() {
            if let Some(p) = pattern {
                if p.is_match(&input) {
                    return false;
                }
            }
        }
        true
    }
}

impl Default for InputSanitizer {
    fn default() -> Self {
        Self::new(true)
    }
}
