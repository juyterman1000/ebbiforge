//! Ebbiforge 'Subconscious' Watcher
//! 
//! Ingests real-world signals into agent states at scale.
//! Triggers "Surprise" ripples for OpenClaw alerts.

use crate::swarm::ProductionTensorSwarm;
use pyo3::prelude::*;

/// A specific data signal (e.g., Price, Log, Metric)
#[pyclass]
#[derive(Clone)]
pub struct Signal {
    #[pyo3(get, set)]
    pub source_id: String,
    #[pyo3(get, set)]
    pub value: f32,
    #[pyo3(get, set)]
    pub surprise_weight: f32,
}

#[pymethods]
impl Signal {
    #[new]
    pub fn new(source_id: String, value: f32, surprise_weight: f32) -> Self {
        Self { source_id, value, surprise_weight }
    }
}

/// The background sensory processor — ingests real-world signals into the swarm
#[pyclass]
pub struct SwarmWatcher {
    swarm: Py<ProductionTensorSwarm>,
    #[pyo3(get, set)]
    pub alert_threshold: f32,
}

#[pymethods]
impl SwarmWatcher {
    #[new]
    pub fn new(swarm: Py<ProductionTensorSwarm>, alert_threshold: f32) -> Self {
        Self {
            swarm,
            alert_threshold,
        }
    }

    /// Ingest a batch of signals and map them to agents
    pub fn ingest_signals(&self, py: Python<'_>, signals: Vec<Signal>) {
        let mut swarm = self.swarm.borrow_mut(py);
        let n = swarm.active.ids.len();
        for (i, signal) in signals.iter().enumerate() {
            if i < n {
                // Map signal to the agent's surprise state proportionally
                swarm.active.surprise_scores[i] = signal.value * signal.surprise_weight;
            }
        }
    }

    /// Check for consensus-based alerts (The "Surprise Ripple")
    pub fn check_for_alerts(&self, py: Python<'_>) -> Vec<String> {
        let swarm = self.swarm.borrow(py);
        let n = swarm.active.ids.len();
        let mut alerts = Vec::new();
        
        // Count how many agents are in a "Surprise Panic" state
        let surprised_count = swarm.active.surprise_scores.iter()
            .filter(|&&s| s > self.alert_threshold)
            .count();
            
        let ratio = if n > 0 { surprised_count as f32 / n as f32 } else { 0.0 };
        
        if ratio > 0.05 { // 5% of the swarm is surprised
            alerts.push(format!("Consensus Anomaly Detected: {:.1}% of swarm in Surprise State", ratio * 100.0));
        }
        
        alerts
    }
}
