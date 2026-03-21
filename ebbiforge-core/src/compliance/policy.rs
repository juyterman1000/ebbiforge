//! Policy Engine for compliance rules
//!
//! Evaluates configurable rules to allow/deny agent actions.

use parking_lot::RwLock;
use pyo3::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use tracing::info;

/// A policy rule
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct Policy {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub action_pattern: String,
    #[pyo3(get)]
    pub allowed: bool,
    #[pyo3(get)]
    pub reason: String,
    #[pyo3(get)]
    pub priority: i32,
}

#[pymethods]
impl Policy {
    #[new]
    pub fn new(id: String, action_pattern: String, allowed: bool, reason: String) -> Self {
        Policy {
            id,
            action_pattern,
            allowed,
            reason,
            priority: 0,
        }
    }

    pub fn __repr__(&self) -> String {
        let status = if self.allowed { "ALLOW" } else { "DENY" };
        format!(
            "Policy({}: {} '{}' -> {})",
            self.id, status, self.action_pattern, self.reason
        )
    }
}

/// Result of policy evaluation
#[derive(Debug, Clone)]
pub struct PolicyResult {
    pub allowed: bool,
    pub policy_id: String,
    pub reason: String,
}

/// Policy engine for rule evaluation
pub struct PolicyEngine {
    policies: RwLock<Vec<Policy>>,
}

impl PolicyEngine {
    pub fn new() -> Self {
        let engine = PolicyEngine {
            policies: RwLock::new(Vec::new()),
        };

        // Add default safety policies
        engine.add_default_policies();
        engine
    }

    fn add_default_policies(&self) {
        // Default deny dangerous actions
        self.add_policy(
            "DENY_DELETE_ALL",
            ".*[Dd]elete.*[Aa]ll.*",
            false,
            "Bulk delete operations require approval",
        );

        self.add_policy(
            "DENY_DROP_TABLE",
            ".*DROP TABLE.*",
            false,
            "Database schema changes require approval",
        );

        self.add_policy(
            "DENY_RM_RF",
            ".*rm -rf.*",
            false,
            "Recursive delete operations blocked",
        );

        self.add_policy(
            "DENY_SUDO",
            ".*sudo.*",
            false,
            "Elevated privilege operations require approval",
        );

        info!("[Policy] Loaded {} default policies", self.count());
    }

    pub fn add_policy(&self, id: &str, action_pattern: &str, allowed: bool, reason: &str) {
        let policy = Policy {
            id: id.to_string(),
            action_pattern: action_pattern.to_string(),
            allowed,
            reason: reason.to_string(),
            priority: 0,
        };

        let mut policies = self.policies.write();
        policies.push(policy);
    }

    pub fn evaluate(&self, _agent_id: &str, action: &str, data: &str) -> PolicyResult {
        let policies = self.policies.read();

        // Combine action and data for pattern matching
        let full_context = format!("{} {}", action, data);

        for policy in policies.iter() {
            if let Ok(re) = Regex::new(&policy.action_pattern) {
                if re.is_match(&full_context) {
                    return PolicyResult {
                        allowed: policy.allowed,
                        policy_id: policy.id.clone(),
                        reason: policy.reason.clone(),
                    };
                }
            }
        }

        // Default allow if no policy matches
        PolicyResult {
            allowed: true,
            policy_id: "DEFAULT_ALLOW".to_string(),
            reason: "No matching policy, default allow".to_string(),
        }
    }

    pub fn count(&self) -> usize {
        self.policies.read().len()
    }

    pub fn list_policies(&self) -> Vec<Policy> {
        self.policies.read().clone()
    }
}

impl Default for PolicyEngine {
    fn default() -> Self {
        Self::new()
    }
}
