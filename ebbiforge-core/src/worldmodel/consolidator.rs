//! Memory Consolidator
//!
//! Compresses old trajectories into summary latent states for long-term memory.

use super::{LatentEncoder, LatentState, WorldModelConfig};
use parking_lot::RwLock;
use pyo3::prelude::*;
use std::collections::HashMap;
use tracing::info;

/// Memory consolidation result
#[derive(Clone)]
#[pyclass]
pub struct ConsolidatedMemory {
    #[pyo3(get)]
    pub summary: LatentState,
    #[pyo3(get)]
    pub num_trajectories: usize,
    #[pyo3(get)]
    pub time_span_hours: f32,
}

#[pymethods]
impl ConsolidatedMemory {
    pub fn __repr__(&self) -> String {
        format!(
            "ConsolidatedMemory(trajectories={}, span={:.1}h)",
            self.num_trajectories, self.time_span_hours
        )
    }
}

/// Memory consolidator for long-term storage
#[pyclass]
pub struct MemoryConsolidator {
    config: WorldModelConfig,
    encoder: LatentEncoder,
    consolidated: RwLock<HashMap<String, Vec<ConsolidatedMemory>>>,
}

#[pymethods]
impl MemoryConsolidator {
    #[new]
    #[pyo3(signature = (config = None))]
    pub fn new(config: Option<WorldModelConfig>) -> Self {
        let cfg = config.clone().unwrap_or_default();
        info!("💾 [Consolidator] Initialized for long-term memory");

        MemoryConsolidator {
            config: cfg.clone(),
            encoder: LatentEncoder::new(config).unwrap(),
            consolidated: RwLock::new(HashMap::new()),
        }
    }

    /// Consolidate multiple trajectories into a single summary
    pub fn consolidate(
        &self,
        agent_id: String,
        trajectory_jsons: Vec<String>,
    ) -> ConsolidatedMemory {
        if trajectory_jsons.is_empty() {
            return ConsolidatedMemory {
                summary: LatentState::new(vec![0.0; self.config.latent_dim], agent_id.clone(), 0),
                num_trajectories: 0,
                time_span_hours: 0.0,
            };
        }

        // Encode each trajectory
        let encoded: Vec<LatentState> = trajectory_jsons
            .iter()
            .map(|tj| self.encoder.encode(tj.clone(), agent_id.clone()))
            .collect();

        // Average the latent vectors using Ebbinghaus exponential decay
        let mut summary_vector = vec![0.0f32; self.config.latent_dim];
        let mut total_weight = 0.0f32;

        // Base the "Now" timestamp off the most recent trajectory in the batch
        let current_time = encoded.iter().map(|s| s.timestamp).max().unwrap_or(0);

        for state in &encoded {
            // Ebbinghaus Formula: Retention = e^(-time_elapsed * decay_rate)
            // But highly surprising events are retained longer!
            // We modulate the decay rate so high surprise = lower decay.
            let hours_elapsed = (current_time.saturating_sub(state.timestamp)) as f32 / 3600.0;
            let effective_decay = self.config.ebbinghaus_decay_rate * (1.0 - state.surprise_score).max(0.1);
            let ebbinghaus_weight = (-hours_elapsed * effective_decay).exp();

            for (i, v) in state.vector.iter().enumerate() {
                summary_vector[i] += v * ebbinghaus_weight;
            }
            total_weight += ebbinghaus_weight;
        }

        if total_weight > 0.0 {
            for v in &mut summary_vector {
                *v /= total_weight;
            }
        }

        // Normalize
        let norm: f32 = summary_vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut summary_vector {
                *v /= norm;
            }
        }

        // Calculate time span
        let timestamps: Vec<u64> = encoded.iter().map(|s| s.timestamp).collect();
        let time_span_hours = if timestamps.len() > 1 {
            let min_t = *timestamps.iter().min().unwrap_or(&0);
            let max_t = *timestamps.iter().max().unwrap_or(&0);
            (max_t - min_t) as f32 / 3600.0
        } else {
            0.0
        };

        let summary = LatentState::new(summary_vector, agent_id.clone(), 0);

        let consolidated = ConsolidatedMemory {
            summary: summary.clone(),
            num_trajectories: trajectory_jsons.len(),
            time_span_hours,
        };

        // Store in consolidated memory
        let mut store = self.consolidated.write();
        let memories = store
            .entry(agent_id.clone())
            .or_insert_with(Vec::new);
        memories.push(consolidated.clone());

        // Forgetting pressure: when capacity exceeded, merge the two lowest-salience
        // memories into one. This is hierarchical compression — old, unsurprising memories
        // get folded together, while salient memories survive longer. Mirrors biological
        // memory reconsolidation where weak traces merge during sleep.
        let max_cap = self.config.max_memories_per_agent;
        if max_cap > 0 {
            while memories.len() > max_cap {
                Self::compress_weakest_pair(memories, self.config.latent_dim);
            }
        }

        info!(
            "💾 [Consolidator] Consolidated {} trajectories for {} ({:.1}h span, {}/{} slots)",
            trajectory_jsons.len(),
            agent_id,
            time_span_hours,
            memories.len(),
            max_cap
        );

        consolidated
    }

    /// Get all consolidated memories for an agent
    pub fn get_memories(&self, agent_id: String) -> Vec<ConsolidatedMemory> {
        let store = self.consolidated.read();
        store.get(&agent_id).cloned().unwrap_or_default()
    }

    /// Merge consolidated memories into a single long-term memory
    pub fn merge_all(&self, agent_id: String) -> LatentState {
        let store = self.consolidated.read();
        let memories = store.get(&agent_id);

        match memories {
            Some(mems) if !mems.is_empty() => {
                let mut merged = vec![0.0f32; self.config.latent_dim];
                let total_weight: f32 = mems.iter().map(|m| m.num_trajectories as f32).sum();

                for mem in mems {
                    let weight = mem.num_trajectories as f32 / total_weight;
                    for (i, v) in mem.summary.vector.iter().enumerate() {
                        merged[i] += v * weight;
                    }
                }

                // Normalize
                let norm: f32 = merged.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for v in &mut merged {
                        *v /= norm;
                    }
                }

                LatentState::new(merged, agent_id, 0)
            }
            _ => LatentState::new(vec![0.0; self.config.latent_dim], agent_id, 0),
        }
    }

    /// Get total trajectory count across all memories
    pub fn total_trajectories(&self, agent_id: String) -> usize {
        let store = self.consolidated.read();
        store
            .get(&agent_id)
            .map(|mems| mems.iter().map(|m| m.num_trajectories).sum())
            .unwrap_or(0)
    }
}

impl MemoryConsolidator {
    /// Find the two lowest-salience memories and merge them into one.
    /// Salience = surprise_score * sqrt(num_trajectories).
    /// Surprising + well-evidenced memories survive; bland ones get folded together.
    fn compress_weakest_pair(memories: &mut Vec<ConsolidatedMemory>, latent_dim: usize) {
        if memories.len() < 2 {
            return;
        }

        let salience = |m: &ConsolidatedMemory| -> f32 {
            m.summary.surprise_score * (m.num_trajectories as f32).sqrt()
        };

        // Find indices of the two lowest-salience memories
        let mut indices: Vec<usize> = (0..memories.len()).collect();
        indices.sort_by(|&a, &b| {
            salience(&memories[a])
                .partial_cmp(&salience(&memories[b]))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let (idx_a, idx_b) = (indices[0], indices[1]);

        // Merge: weighted average by trajectory count
        let (lo, hi) = if idx_a < idx_b { (idx_a, idx_b) } else { (idx_b, idx_a) };
        let total_traj = memories[lo].num_trajectories + memories[hi].num_trajectories;
        let w_a = memories[lo].num_trajectories as f32 / total_traj as f32;
        let w_b = memories[hi].num_trajectories as f32 / total_traj as f32;

        let mut merged_vec = vec![0.0f32; latent_dim];
        for j in 0..latent_dim.min(memories[lo].summary.vector.len()) {
            merged_vec[j] = memories[lo].summary.vector[j] * w_a
                          + memories[hi].summary.vector[j] * w_b;
        }

        // Re-normalize to unit sphere
        let norm: f32 = merged_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut merged_vec {
                *v /= norm;
            }
        }

        // Merged surprise: weighted average
        let merged_surprise = memories[lo].summary.surprise_score * w_a
                            + memories[hi].summary.surprise_score * w_b;
        let merged_span = memories[lo].time_span_hours.max(memories[hi].time_span_hours);

        let mut merged_state = LatentState::new(
            merged_vec,
            memories[lo].summary.agent_id.clone(),
            0,
        );
        merged_state.surprise_score = merged_surprise;

        let merged = ConsolidatedMemory {
            summary: merged_state,
            num_trajectories: total_traj,
            time_span_hours: merged_span,
        };

        // Remove higher index first to preserve lower index validity
        memories.remove(hi);
        memories.remove(lo);
        memories.push(merged);
    }
}
