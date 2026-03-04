use crate::core::middleware::{EbbiforgeContext, Middleware};
use crate::TrajectoryPoint;
use parking_lot::RwLock;
use pyo3::prelude::*;
use std::collections::HashMap;
use tracing::warn;

/// A predictive safety mechanism that evaluates agent intentions against
/// historical failure patterns.
///
/// `PredictiveSafetyShield` prevents high-risk actions from reaching execution
/// by analyzing the trajectory similarity to known failure modes.
#[pyclass]
pub struct PredictiveSafetyShield {
    /// In-memory cache of identified failure trajectories
    failure_trajectories: RwLock<HashMap<String, Vec<TrajectoryPoint>>>,
    /// The threshold score [0.0 - 1.0] above which an action is blocked
    risk_threshold: f64,
}

#[pymethods]
impl PredictiveSafetyShield {
    /// Initializes the safety shield with a specified risk sensitivity.
    #[new]
    pub fn new(risk_threshold: f64) -> Self {
        PredictiveSafetyShield {
            failure_trajectories: RwLock::new(HashMap::new()),
            risk_threshold,
        }
    }

    /// Ingests a labeled failure trajectory for use in pattern matching.
    pub fn add_failure_pattern(&self, id: String, trajectory_json: String) {
        if let Ok(traj) = serde_json::from_str::<Vec<TrajectoryPoint>>(&trajectory_json) {
            let mut failures = self.failure_trajectories.write();
            failures.insert(id, traj);
        }
    }

    /// Get the number of loaded failure patterns
    pub fn pattern_count(&self) -> usize {
        let failures = self.failure_trajectories.read();
        failures.len()
    }

    /// Evaluates the risk level of the current agent trajectory.
    ///
    /// Returns a tuple containing the risk score [0.0 - 1.0] and an optional
    /// reason string if a high-similarity failure mode is detected.
    pub fn analyze_risk(&self, trajectory_json: String) -> (f64, Option<String>) {
        let current: Vec<TrajectoryPoint> =
            serde_json::from_str(&trajectory_json).unwrap_or_default();
        let failures = self.failure_trajectories.read();

        if failures.is_empty() {
            return (0.0, None);
        }

        let mut max_risk = 0.0;
        let mut danger_reason: Option<String> = None;

        for (id, failure_path) in failures.iter() {
            let similarity = calculate_similarity(&current, failure_path);
            if similarity > max_risk {
                max_risk = similarity;
                let short_id = if id.len() > 8 { &id[..8] } else { id };
                danger_reason = Some(format!("Pattern matches historical failure {}", short_id));
            }
        }

        (max_risk, danger_reason)
    }
}

/// Internal helper function (not exposed to Python)
fn calculate_similarity(current: &[TrajectoryPoint], historical: &[TrajectoryPoint]) -> f64 {
    let compare_len = current.len().min(historical.len());
    if compare_len == 0 {
        return 0.0;
    }

    let mut matches = 0;
    for i in 0..compare_len {
        if current[i].action == historical[i].action {
            matches += 1;
        }
    }

    matches as f64 / historical.len() as f64
}

impl Middleware for PredictiveSafetyShield {
    fn name(&self) -> &str {
        "PredictiveSafetyShield"
    }

    fn before_step(&self, ctx: &mut EbbiforgeContext) -> Result<(), String> {
        let trajectory_json = ctx.get_trajectory_json();
        let (risk_score, reason) = self.analyze_risk(trajectory_json);

        if risk_score >= self.risk_threshold {
            warn!("[SafetyShield] BLOCKING EXECUTION. Risk: {:.2}", risk_score);
            if let Some(ref r) = reason {
                warn!("   Reason: {}", r);
            }

            ctx.should_stop = true;
            ctx.stop_reason = Some(format!("Safety Block: {}", reason.unwrap_or_default()));
        }

        Ok(())
    }
}
