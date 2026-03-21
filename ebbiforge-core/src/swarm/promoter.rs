//! Promotion Logic
//!
//! Logic to "promote" a Light Agent (swarm) to a Heavy Agent (LLM).
//! Triggered by conflict or complexity thresholds.

use super::tensor_engine::TensorSwarm;
use pyo3::prelude::*;
use tracing::info;

/// Logic for promoting agents
#[pyclass]
pub struct PromotionLogic {
    conflict_threshold: f32,
}

#[pymethods]
impl PromotionLogic {
    #[new]
    pub fn new() -> Self {
        PromotionLogic {
            conflict_threshold: 0.5, // 50% max interaction
        }
    }

    /// Check which agents need promotion based on density exceeding the conflict threshold.
    /// Agents in crowded cells are promoted to Heavy (LLM) for complex decision-making.
    pub fn find_promotion_candidates(
        &self,
        swarm: &TensorSwarm,
        density_map: &super::GridMap,
    ) -> Vec<u32> {
        let mut candidates = Vec::new();

        // Check agents using conflict_threshold for density comparison
        let scan_limit = 100.min(swarm.ids.len());
        for i in 0..scan_limit {
            let x = swarm.x[i];
            let y = swarm.y[i];
            let density = density_map.get_density(x, y);

            // Promote if density exceeds conflict threshold (config-driven)
            if density as f32 > self.conflict_threshold * 10.0 {
                candidates.push(swarm.ids[i]);
            }
        }

        if !candidates.is_empty() {
            info!(
                "🚀 [Promoter] Promoting {} agents to Heavy (LLM) status due to conflict.",
                candidates.len()
            );
        }

        candidates
    }

    /// Inflate a Light Agent struct into a full Agent prompt context
    pub fn inflate_context(&self, agent_id: u32, state: (f32, f32, f32)) -> String {
        format!(
            "You are Agent #{}. Status: Health={:.2}, Pos=({:.1}, {:.1}). You have been promoted to handle a local conflict.",
            agent_id, state.2, state.0, state.1
        )
    }
}
