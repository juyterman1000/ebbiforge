//! Audit logging for compliance
//!
//! Provides immutable, append-only audit trail for all agent actions.

use parking_lot::RwLock;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::warn; // Add tracing::warn for logging

/// Single audit event
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct AuditEvent {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub timestamp: String,
    #[pyo3(get)]
    pub agent_id: String,
    #[pyo3(get)]
    pub action: String,
    #[pyo3(get)]
    pub outcome: String, // "APPROVED" or "DENIED"
    #[pyo3(get)]
    pub reason: Option<String>,
}

#[pymethods]
impl AuditEvent {
    pub fn __repr__(&self) -> String {
        format!(
            "[{}] {} {} -> {}",
            self.timestamp, self.agent_id, self.action, self.outcome
        )
    }
}

/// Thread-safe audit logger
pub struct AuditLogger {
    events: RwLock<Vec<AuditEvent>>,
    user_index: RwLock<HashMap<String, Vec<usize>>>, // user_id -> event indices
}

impl AuditLogger {
    pub fn new() -> Self {
        AuditLogger {
            events: RwLock::new(Vec::new()),
            user_index: RwLock::new(HashMap::new()),
        }
    }

    fn generate_id() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        format!("audit-{}", nanos)
    }

    fn now() -> String {
        // Simple timestamp without chrono
        use std::time::{SystemTime, UNIX_EPOCH};
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        format!("{}", secs)
    }

    pub fn log_approval(&self, agent_id: &str, action: &str) -> String {
        let event = AuditEvent {
            id: Self::generate_id(),
            timestamp: Self::now(),
            agent_id: agent_id.to_string(),
            action: action.to_string(),
            outcome: "APPROVED".to_string(),
            reason: None,
        };
        let id = event.id.clone();

        let mut events = self.events.write();
        let idx = events.len();
        events.push(event);

        // Index by agent_id for per-agent audit retrieval
        let mut index = self.user_index.write();
        index.entry(agent_id.to_string()).or_default().push(idx);

        id
    }

    pub fn log_denial(&self, agent_id: &str, action: &str, reason: &str) -> String {
        let event = AuditEvent {
            id: Self::generate_id(),
            timestamp: Self::now(),
            agent_id: agent_id.to_string(),
            action: action.to_string(),
            outcome: "DENIED".to_string(),
            reason: Some(reason.to_string()),
        };
        let id = event.id.clone();

        let mut events = self.events.write();
        let idx = events.len();
        events.push(event);

        let mut index = self.user_index.write();
        index.entry(agent_id.to_string()).or_default().push(idx);

        warn!("🚫 [Audit] DENIED: {} -> {} ({})", agent_id, action, reason);
        id
    }

    pub fn export_json(&self) -> String {
        let events = self.events.read();
        serde_json::to_string_pretty(&*events).unwrap_or_default()
    }

    pub fn export_user_logs(&self, user_id: &str) -> String {
        let events = self.events.read();
        let index = self.user_index.read();

        let user_events: Vec<&AuditEvent> = index
            .get(user_id)
            .map(|indices| indices.iter().filter_map(|&i| events.get(i)).collect())
            .unwrap_or_default();

        serde_json::to_string_pretty(&user_events).unwrap_or_default()
    }

    pub fn delete_user_logs(&self, user_id: &str) {
        let mut index = self.user_index.write();
        index.remove(user_id);
        // Note: Actual events not deleted for audit immutability, but index removed
    }

    pub fn count(&self) -> usize {
        self.events.read().len()
    }
}

impl Default for AuditLogger {
    fn default() -> Self {
        Self::new()
    }
}
