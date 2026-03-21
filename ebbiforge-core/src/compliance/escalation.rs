//! Escalation Flow for human-in-the-loop approval
//!
//! Queues high-risk actions for human approval before execution.

use parking_lot::RwLock;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tracing::{info, warn};

/// Risk level for actions
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[pyclass]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[pymethods]
impl RiskLevel {
    pub fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

/// An action pending approval
#[derive(Clone, Debug, Serialize, Deserialize)]
#[pyclass]
pub struct PendingAction {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub agent_id: String,
    #[pyo3(get)]
    pub action: String,
    #[pyo3(get)]
    pub data: String,
    #[pyo3(get)]
    pub risk_level: String,
    #[pyo3(get)]
    pub reason: String,
    #[pyo3(get)]
    pub timestamp: u64,
}

#[pymethods]
impl PendingAction {
    pub fn __repr__(&self) -> String {
        format!(
            "PendingAction({}: {} -> {} [{}])",
            self.id, self.agent_id, self.action, self.risk_level
        )
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

/// Result of escalation check
#[derive(Clone)]
#[pyclass]
pub struct EscalationResult {
    #[pyo3(get)]
    pub needs_approval: bool,
    #[pyo3(get)]
    pub risk_level: String,
    #[pyo3(get)]
    pub reason: String,
    #[pyo3(get)]
    pub pending_id: Option<String>,
}

#[pymethods]
impl EscalationResult {
    pub fn __repr__(&self) -> String {
        if self.needs_approval {
            format!(
                "EscalationResult(PENDING: {} [{}])",
                self.reason, self.risk_level
            )
        } else {
            "EscalationResult(AUTO_APPROVED)".to_string()
        }
    }
}

/// Escalation flow manager
#[pyclass]
pub struct EscalationFlow {
    pending_queue: RwLock<VecDeque<PendingAction>>,
    high_risk_patterns: Vec<String>,
    critical_patterns: Vec<String>,
}

#[pymethods]
impl EscalationFlow {
    #[new]
    pub fn new() -> Self {
        let flow = EscalationFlow {
            pending_queue: RwLock::new(VecDeque::new()),
            high_risk_patterns: vec![
                "delete".to_string(),
                "drop".to_string(),
                "remove".to_string(),
                "send_email".to_string(),
                "transfer".to_string(),
                "payment".to_string(),
            ],
            critical_patterns: vec![
                "sudo".to_string(),
                "rm -rf".to_string(),
                "format".to_string(),
                "shutdown".to_string(),
                "api_key".to_string(),
                "password".to_string(),
                "credential".to_string(),
            ],
        };
        info!(
            "🚨 [Escalation] Initialized with {} high-risk, {} critical patterns",
            flow.high_risk_patterns.len(),
            flow.critical_patterns.len()
        );
        flow
    }

    /// Check if an action needs escalation
    pub fn check(&self, agent_id: String, action: String, data: String) -> EscalationResult {
        let action_lower = action.to_lowercase();
        let data_lower = data.to_lowercase();
        let combined = format!("{} {}", action_lower, data_lower);

        // Check for critical patterns
        for pattern in &self.critical_patterns {
            if combined.contains(pattern) {
                let pending = self.queue_action(
                    &agent_id,
                    &action,
                    &data,
                    "Critical",
                    &format!("Contains critical pattern: {}", pattern),
                );
                return EscalationResult {
                    needs_approval: true,
                    risk_level: "Critical".to_string(),
                    reason: format!("Action contains critical pattern: {}", pattern),
                    pending_id: Some(pending.id),
                };
            }
        }

        // Check for high-risk patterns
        for pattern in &self.high_risk_patterns {
            if combined.contains(pattern) {
                let pending = self.queue_action(
                    &agent_id,
                    &action,
                    &data,
                    "High",
                    &format!("Contains high-risk pattern: {}", pattern),
                );
                return EscalationResult {
                    needs_approval: true,
                    risk_level: "High".to_string(),
                    reason: format!("Action contains high-risk pattern: {}", pattern),
                    pending_id: Some(pending.id),
                };
            }
        }

        // Auto-approve low-risk actions
        EscalationResult {
            needs_approval: false,
            risk_level: "Low".to_string(),
            reason: String::new(),
            pending_id: None,
        }
    }

    /// Get all pending actions
    pub fn get_pending(&self) -> Vec<PendingAction> {
        let queue = self.pending_queue.read();
        queue.iter().cloned().collect()
    }

    /// Approve a pending action
    pub fn approve(&self, pending_id: String) -> bool {
        let mut queue = self.pending_queue.write();
        if let Some(pos) = queue.iter().position(|p| p.id == pending_id) {
            if let Some(action) = queue.remove(pos) {
                info!(
                    "[Escalation] Approved: {} -> {}",
                    action.agent_id, action.action
                );
                return true;
            }
            false
        } else {
            false
        }
    }

    /// Reject a pending action
    pub fn reject(&self, pending_id: String) -> bool {
        let mut queue = self.pending_queue.write();
        if let Some(pos) = queue.iter().position(|p| p.id == pending_id) {
            if let Some(action) = queue.remove(pos) {
                warn!(
                    "[Escalation] Rejected: {} -> {}",
                    action.agent_id, action.action
                );
                return true;
            }
            false
        } else {
            false
        }
    }

    /// Get queue size
    pub fn pending_count(&self) -> usize {
        self.pending_queue.read().len()
    }
}

impl EscalationFlow {
    fn queue_action(
        &self,
        agent_id: &str,
        action: &str,
        data: &str,
        risk: &str,
        reason: &str,
    ) -> PendingAction {
        let pending = PendingAction {
            id: format!(
                "esc-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
            ),
            agent_id: agent_id.to_string(),
            action: action.to_string(),
            data: data.to_string(),
            risk_level: risk.to_string(),
            reason: reason.to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        let mut queue = self.pending_queue.write();
        queue.push_back(pending.clone());
        warn!(
            "🚨 [Escalation] Queued for approval: {} -> {} [{}]",
            agent_id, action, risk
        );
        pending
    }
}

impl Default for EscalationFlow {
    fn default() -> Self {
        Self::new()
    }
}
