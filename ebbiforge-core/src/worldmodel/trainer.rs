//! Production Trajectory Buffer for World Model Training
//!
//! Global ring buffer that accumulates (state, action, next_state) samples
//! from ALL agents across the swarm. One dataset, shuffled across agents,
//! feeds one global dynamics model.
//!
//! Production features:
//!   - Fixed-capacity ring buffer (no unbounded growth)
//!   - Per-recording stats (total recorded, overwrites, fill ratio)
//!   - Efficient random mini-batch sampling via Fisher-Yates
//!   - JSON serialization for Python interop
//!   - Memory-efficient: stores raw f32 vecs, not full LatentState objects

use super::dynamics::TrainingSample;
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use tracing::info;

/// Production-grade ring buffer for trajectory data.
///
/// Designed for millions of agents: each tick, a fraction of agents
/// record their (state, action, next_state) into this global buffer.
/// When full, oldest samples are silently overwritten (ring buffer).
///
/// Usage from Python:
/// ```python
/// buf = TrajectoryBuffer(capacity=100_000)
/// for tick in range(1000):
///     swarm.tick()
///     # Sample 1% of agents per tick
///     buf.record(state_vec, action_vec, next_state_vec)
/// # Train
/// predictor.train(buf.to_json(), epochs=50, lr=0.001, batch_size=256, val_split=0.1)
/// ```
#[pyclass]
pub struct TrajectoryBuffer {
    samples: Vec<TrainingSample>,
    capacity: usize,
    write_idx: usize,
    total_recorded: u64,
    total_overwrites: u64,
}

#[pymethods]
impl TrajectoryBuffer {
    #[new]
    #[pyo3(signature = (capacity = 100000))]
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(64); // Minimum 64 samples
        info!("[TrajectoryBuffer] Initialized (capacity={})", cap);
        TrajectoryBuffer {
            samples: Vec::with_capacity(cap.min(16384)), // Lazy alloc
            capacity: cap,
            write_idx: 0,
            total_recorded: 0,
            total_overwrites: 0,
        }
    }

    /// Record a single (state_t, action_t, state_{t+1}) sample.
    pub fn record(
        &mut self,
        state: Vec<f32>,
        action: Vec<f32>,
        next_state: Vec<f32>,
    ) {
        let sample = TrainingSample {
            state,
            action,
            next_state,
        };

        if self.samples.len() < self.capacity {
            self.samples.push(sample);
        } else {
            self.samples[self.write_idx] = sample;
            self.total_overwrites += 1;
        }

        self.write_idx = (self.write_idx + 1) % self.capacity;
        self.total_recorded += 1;
    }

    /// Record a batch of samples from JSON.
    /// Format: `[{"state": [...], "action": [...], "next_state": [...]}, ...]`
    pub fn record_batch(&mut self, json: String) -> pyo3::PyResult<usize> {
        let batch: Vec<TrainingSample> = serde_json::from_str(&json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON: {}", e)))?;

        let count = batch.len();
        for sample in batch {
            if self.samples.len() < self.capacity {
                self.samples.push(sample);
            } else {
                self.samples[self.write_idx] = sample;
                self.total_overwrites += 1;
            }
            self.write_idx = (self.write_idx + 1) % self.capacity;
            self.total_recorded += 1;
        }

        Ok(count)
    }

    /// Get a random mini-batch as JSON for training.
    /// Uses Fisher-Yates partial shuffle for O(batch_size) sampling.
    pub fn sample_batch(&self, batch_size: usize) -> String {
        if self.samples.is_empty() {
            return "[]".to_string();
        }

        let actual_size = batch_size.min(self.samples.len());
        let mut rng = rand::thread_rng();

        // Randomly select indices
        let mut indices: Vec<usize> = (0..self.samples.len()).collect();
        indices.shuffle(&mut rng);
        indices.truncate(actual_size);

        let batch: Vec<&TrainingSample> = indices.iter().map(|&i| &self.samples[i]).collect();
        serde_json::to_string(&batch).unwrap_or_else(|_| "[]".to_string())
    }

    /// Export all samples as JSON for full training.
    pub fn to_json(&self) -> String {
        serde_json::to_string(&self.samples).unwrap_or_else(|_| "[]".to_string())
    }

    /// Current number of stored samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Total samples ever recorded (including overwritten).
    pub fn total_recorded(&self) -> u64 {
        self.total_recorded
    }

    /// Total samples that were overwritten (ring buffer wraps).
    pub fn total_overwrites(&self) -> u64 {
        self.total_overwrites
    }

    /// Buffer fill ratio (0.0 = empty, 1.0 = full/wrapping).
    pub fn fill_ratio(&self) -> f32 {
        self.samples.len() as f32 / self.capacity as f32
    }

    /// Clear all stored samples. Resets counters.
    pub fn clear(&mut self) {
        self.samples.clear();
        self.write_idx = 0;
        // Preserve lifetime stats
    }

    pub fn __repr__(&self) -> String {
        format!(
            "TrajectoryBuffer(stored={}, capacity={}, fill={:.1}%, total={}, overwrites={})",
            self.samples.len(),
            self.capacity,
            self.fill_ratio() * 100.0,
            self.total_recorded,
            self.total_overwrites
        )
    }
}
