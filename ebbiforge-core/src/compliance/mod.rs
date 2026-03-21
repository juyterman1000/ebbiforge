//! Compliance module for Regulated AI
//!
//! This module provides enterprise-grade compliance features:
//! - Audit logging (immutable action records)
//! - PII redaction (auto-detect and mask sensitive data)
//! - Policy enforcement (configurable rules engine)
//! - Decision traceability (full lineage tracking)
//! - Rate limiting (cost control)
//! - Escalation flow (human-in-the-loop)
//! - Input sanitization (prompt injection defense)

pub mod audit;
pub mod escalation;
pub mod pii;
pub mod policy;
pub mod ratelimit;
pub mod sanitizer;
pub mod trace;

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::info;

pub use audit::AuditLogger;
pub use escalation::EscalationFlow;
pub use pii::PIIRedactor;
pub use policy::PolicyEngine;
pub use ratelimit::RateLimiter;
pub use sanitizer::InputSanitizer;
pub use trace::DecisionTracker;

/// Result of a compliance check
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct ComplianceResult {
    #[pyo3(get)]
    pub approved: bool,
    #[pyo3(get)]
    pub reason: String,
    #[pyo3(get)]
    pub policy_id: Option<String>,
    #[pyo3(get)]
    pub audit_id: String,
}

#[pymethods]
impl ComplianceResult {
    #[new]
    #[pyo3(signature = (approved, reason, policy_id = None, audit_id = "".to_string()))]
    pub fn new(
        approved: bool,
        reason: String,
        policy_id: Option<String>,
        audit_id: String,
    ) -> Self {
        ComplianceResult {
            approved,
            reason,
            policy_id,
            audit_id,
        }
    }

    pub fn __repr__(&self) -> String {
        if self.approved {
            format!("ComplianceResult(APPROVED, audit_id='{}')", self.audit_id)
        } else {
            format!(
                "ComplianceResult(DENIED: '{}', policy='{:?}')",
                self.reason, self.policy_id
            )
        }
    }
}

/// Main compliance engine that orchestrates all checks
#[pyclass]
pub struct ComplianceEngine {
    audit_logger: Arc<AuditLogger>,
    pii_redactor: Arc<PIIRedactor>,
    policy_engine: Arc<PolicyEngine>,
    decision_tracker: Arc<DecisionTracker>,
}

#[pymethods]
impl ComplianceEngine {
    #[new]
    pub fn new() -> Self {
        info!("⚖️  [Compliance] Engine initialized");
        ComplianceEngine {
            audit_logger: Arc::new(AuditLogger::new()),
            pii_redactor: Arc::new(PIIRedactor::default()),
            policy_engine: Arc::new(PolicyEngine::new()),
            decision_tracker: Arc::new(DecisionTracker::new()),
        }
    }

    /// Check if an action is allowed by policy
    pub fn check_action(&self, agent_id: String, action: String, data: String) -> ComplianceResult {
        // Start decision trace
        let trace_id = self.decision_tracker.start_trace(&agent_id, &action);

        // Check for PII in data
        let pii_detected = self.pii_redactor.detect_pii(&data);
        if !pii_detected.is_empty() {
            let reason = format!("PII detected: {:?}", pii_detected);
            self.audit_logger.log_denial(&agent_id, &action, &reason);
            return ComplianceResult {
                approved: false,
                reason,
                policy_id: Some("PII_PROTECTION".to_string()),
                audit_id: trace_id,
            };
        }

        // Evaluate policy rules
        let policy_result = self.policy_engine.evaluate(&agent_id, &action, &data);

        if policy_result.allowed {
            self.audit_logger.log_approval(&agent_id, &action);
            ComplianceResult {
                approved: true,
                reason: "Action approved".to_string(),
                policy_id: None,
                audit_id: trace_id,
            }
        } else {
            self.audit_logger
                .log_denial(&agent_id, &action, &policy_result.reason);
            ComplianceResult {
                approved: false,
                reason: policy_result.reason,
                policy_id: Some(policy_result.policy_id),
                audit_id: trace_id,
            }
        }
    }

    /// Redact PII from text
    pub fn redact_pii(&self, text: String) -> String {
        self.pii_redactor.redact(&text)
    }

    /// Add a policy rule
    pub fn add_policy(
        &self,
        policy_id: String,
        action_pattern: String,
        allowed: bool,
        reason: String,
    ) {
        self.policy_engine
            .add_policy(&policy_id, &action_pattern, allowed, &reason);
    }

    /// Export audit logs as JSON
    pub fn export_audit_logs(&self) -> String {
        self.audit_logger.export_json()
    }

    /// Get decision trace for an audit ID
    pub fn get_trace(&self, trace_id: String) -> String {
        self.decision_tracker.get_trace(&trace_id)
    }

    /// GDPR: Delete all data for a user
    pub fn delete_user_data(&self, user_id: String) -> bool {
        self.audit_logger.delete_user_logs(&user_id);
        self.decision_tracker.delete_user_traces(&user_id);
        info!("🗑️  [GDPR] Deleted all data for user: {}", user_id);
        true
    }

    /// GDPR: Export all data for a user
    pub fn export_user_data(&self, user_id: String) -> String {
        let logs = self.audit_logger.export_user_logs(&user_id);
        let traces = self.decision_tracker.export_user_traces(&user_id);
        serde_json::json!({
            "user_id": user_id,
            "audit_logs": logs,
            "decision_traces": traces
        })
        .to_string()
    }

    /// Get compliance statistics
    pub fn stats(&self) -> String {
        let audit_count = self.audit_logger.count();
        let policy_count = self.policy_engine.count();
        let trace_count = self.decision_tracker.count();
        format!(
            "ComplianceStats(audits={}, policies={}, traces={})",
            audit_count, policy_count, trace_count
        )
    }
}

impl Default for ComplianceEngine {
    fn default() -> Self {
        Self::new()
    }
}
