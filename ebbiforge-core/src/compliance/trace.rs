//! Decision traceability for compliance
//!
//! Tracks the full lineage of agent decisions for audit purposes.

use parking_lot::RwLock;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single step in a decision trace
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct TraceStep {
    #[pyo3(get)]
    pub step_id: String,
    #[pyo3(get)]
    pub timestamp: String,
    #[pyo3(get)]
    pub action: String,
    #[pyo3(get)]
    pub input: String,
    #[pyo3(get)]
    pub output: String,
    #[pyo3(get)]
    pub duration_ms: u64,
}

/// A complete decision trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTrace {
    pub trace_id: String,
    pub agent_id: String,
    pub started_at: String,
    pub steps: Vec<TraceStep>,
    pub status: String, // "running", "completed", "failed"
}

/// Decision tracker for full lineage
pub struct DecisionTracker {
    traces: RwLock<HashMap<String, DecisionTrace>>,
    user_index: RwLock<HashMap<String, Vec<String>>>, // user_id -> trace_ids
}

impl DecisionTracker {
    pub fn new() -> Self {
        DecisionTracker {
            traces: RwLock::new(HashMap::new()),
            user_index: RwLock::new(HashMap::new()),
        }
    }

    fn generate_id(prefix: &str) -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        format!("{}-{}", prefix, nanos)
    }

    fn now() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        format!("{}", secs)
    }

    pub fn start_trace(&self, agent_id: &str, action: &str) -> String {
        let trace_id = Self::generate_id("trace");

        let trace = DecisionTrace {
            trace_id: trace_id.clone(),
            agent_id: agent_id.to_string(),
            started_at: Self::now(),
            steps: vec![TraceStep {
                step_id: Self::generate_id("step"),
                timestamp: Self::now(),
                action: action.to_string(),
                input: "".to_string(),
                output: "".to_string(),
                duration_ms: 0,
            }],
            status: "running".to_string(),
        };

        let mut traces = self.traces.write();
        traces.insert(trace_id.clone(), trace);

        // Index by agent_id
        let mut index = self.user_index.write();
        index
            .entry(agent_id.to_string())
            .or_default()
            .push(trace_id.clone());

        trace_id
    }

    pub fn add_step(
        &self,
        trace_id: &str,
        action: &str,
        input: &str,
        output: &str,
        duration_ms: u64,
    ) {
        let mut traces = self.traces.write();
        if let Some(trace) = traces.get_mut(trace_id) {
            trace.steps.push(TraceStep {
                step_id: Self::generate_id("step"),
                timestamp: Self::now(),
                action: action.to_string(),
                input: input.to_string(),
                output: output.to_string(),
                duration_ms,
            });
        }
    }

    pub fn complete_trace(&self, trace_id: &str, status: &str) {
        let mut traces = self.traces.write();
        if let Some(trace) = traces.get_mut(trace_id) {
            trace.status = status.to_string();
        }
    }

    pub fn get_trace(&self, trace_id: &str) -> String {
        let traces = self.traces.read();
        traces
            .get(trace_id)
            .map(|t| serde_json::to_string_pretty(t).unwrap_or_default())
            .unwrap_or_else(|| "{}".to_string())
    }

    pub fn get_lineage(&self, trace_id: &str) -> Vec<TraceStep> {
        let traces = self.traces.read();
        traces
            .get(trace_id)
            .map(|t| t.steps.clone())
            .unwrap_or_default()
    }

    pub fn export_user_traces(&self, user_id: &str) -> String {
        let traces = self.traces.read();
        let index = self.user_index.read();

        let user_traces: Vec<&DecisionTrace> = index
            .get(user_id)
            .map(|ids| ids.iter().filter_map(|id| traces.get(id)).collect())
            .unwrap_or_default();

        serde_json::to_string_pretty(&user_traces).unwrap_or_default()
    }

    pub fn delete_user_traces(&self, user_id: &str) {
        let mut index = self.user_index.write();
        if let Some(trace_ids) = index.remove(user_id) {
            let mut traces = self.traces.write();
            for id in trace_ids {
                traces.remove(&id);
            }
        }
    }

    pub fn count(&self) -> usize {
        self.traces.read().len()
    }
}

impl Default for DecisionTracker {
    fn default() -> Self {
        Self::new()
    }
}
