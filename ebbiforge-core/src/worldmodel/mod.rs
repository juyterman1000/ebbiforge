//! Latent World Model for Ebbiforge
//!
//! Provides predictive capabilities for agent planning:
//! - Latent encoding (compress context to fixed-size vector)
//! - Autoregressive dynamics (predict next state)
//! - Planning engine (rollout futures, pick best action)
//! - Memory consolidation (compress old trajectories)
//!
//! Based on:
//! - "Latent-Space Predictive dynamics"
//! - "Latent Diffusion dynamics"

pub mod consolidator;
pub mod diffusion;
pub mod dynamics;
pub mod encoder;
pub mod geometric;
pub mod planner;
pub mod trainer;

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

pub use consolidator::MemoryConsolidator;
pub use diffusion::DiffusionPredictor;
pub use dynamics::AutoregressivePredictor;
pub use encoder::LatentEncoder;
pub use geometric::GeometricEncoder;
pub use planner::PlanningEngine;
pub use trainer::TrajectoryBuffer;

/// Latent state representation (compressed context)
#[derive(Clone, Debug, Serialize, Deserialize)]
#[pyclass]
pub struct LatentState {
    #[pyo3(get)]
    pub vector: Vec<f32>,
    #[pyo3(get)]
    pub timestamp: u64,
    #[pyo3(get)]
    pub agent_id: String,
    #[pyo3(get)]
    pub step: u32,
    /// Emotion proxy: Prediction Error (KL/Cosine divergence from predicted reality)
    #[pyo3(get)]
    pub surprise_score: f32,
}

#[pymethods]
impl LatentState {
    #[new]
    pub fn new(vector: Vec<f32>, agent_id: String, step: u32) -> Self {
        LatentState {
            vector,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            agent_id,
            step,
            surprise_score: 0.0, // Default to no surprise until computed
        }
    }

    pub fn similarity(&self, other: &LatentState) -> f32 {
        if self.vector.len() != other.vector.len() {
            return 0.0;
        }

        let dot: f32 = self
            .vector
            .iter()
            .zip(&other.vector)
            .map(|(a, b)| a * b)
            .sum();
        let norm_a: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "LatentState(dim={}, agent={}, step={}, surprise={:.3})",
            self.vector.len(),
            self.agent_id,
            self.step,
            self.surprise_score
        )
    }

    /// Calculate mathematically how "surprised" this state is compared to what was predicted.
    /// Uses Cosine Similarity. 0.0 = Totally boring/predictable. 1.0 = Complete anomaly.
    pub fn compute_surprise(&mut self, predicted_prior: &LatentState) {
        // Assertion: ensure both vectors are L2-normalized before computing cosine similarity
        let norm_self: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_pred: f32 = predicted_prior.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        debug_assert!(
            (norm_self - 1.0).abs() < 1e-3 || norm_self == 0.0,
            "Real state not properly L2-normalized! Norm: {}", norm_self
        );
        debug_assert!(
            (norm_pred - 1.0).abs() < 1e-3 || norm_pred == 0.0,
            "Predicted state not properly L2-normalized! Norm: {}", norm_pred
        );

        let sim = self.similarity(predicted_prior);
        // Ensure similarity is clamped between -1.0 and 1.0
        let clamped_sim = sim.max(-1.0).min(1.0);
        // Map Cosine (-1.0 to 1.0) into Surprise (1.0 to 0.0)
        self.surprise_score = (1.0 - clamped_sim) / 2.0;
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

/// World model configuration
#[derive(Clone, Debug)]
#[pyclass]
pub struct WorldModelConfig {
    #[pyo3(get, set)]
    pub latent_dim: usize,
    #[pyo3(get, set)]
    pub context_window: usize,
    #[pyo3(get, set)]
    pub prediction_steps: usize,
    #[pyo3(get, set)]
    pub learning_rate: f32,
    
    /// Base decay rate for Ebbinghaus retention curve (how fast memories fade)
    #[pyo3(get, set)]
    pub ebbinghaus_decay_rate: f32,

    /// Maximum consolidated memories per agent before forgetting pressure kicks in.
    /// When exceeded, the two lowest-salience memories are merged into one.
    #[pyo3(get, set)]
    pub max_memories_per_agent: usize,

    #[pyo3(get, set)]
    pub grid_size: (usize, usize),
}

#[pymethods]
impl WorldModelConfig {
    #[new]
    #[pyo3(signature = (latent_dim = 768, context_window = 8, prediction_steps = 4, learning_rate = 0.001, ebbinghaus_decay_rate = 0.1, max_memories_per_agent = 64, grid_size = (100, 100)))]
    pub fn new(
        latent_dim: usize,
        context_window: usize,
        prediction_steps: usize,
        learning_rate: f32,
        ebbinghaus_decay_rate: f32,
        max_memories_per_agent: usize,
        grid_size: (usize, usize),
    ) -> Self {
        WorldModelConfig {
            latent_dim,
            context_window,
            prediction_steps,
            learning_rate,
            ebbinghaus_decay_rate,
            max_memories_per_agent,
            grid_size,
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "WorldModelConfig(dim={}, window={}, steps={})",
            self.latent_dim, self.context_window, self.prediction_steps
        )
    }
}

impl Default for WorldModelConfig {
    fn default() -> Self {
        Self::new(768, 8, 4, 0.001, 0.1, 64, (100, 100))
    }
}

/// Prediction result from world model
#[derive(Clone, Debug, Serialize, Deserialize)]
#[pyclass]
pub struct Prediction {
    #[pyo3(get)]
    pub future_states: Vec<LatentState>,
    #[pyo3(get)]
    pub confidence: f32,
    #[pyo3(get)]
    pub action_sequence: Vec<String>,
}

#[pymethods]
impl Prediction {
    #[new]
    pub fn new(
        future_states: Vec<LatentState>,
        confidence: f32,
        action_sequence: Vec<String>,
    ) -> Self {
        Prediction {
            future_states,
            confidence,
            action_sequence,
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Prediction(steps={}, confidence={:.2}, actions={:?})",
            self.future_states.len(),
            self.confidence,
            self.action_sequence
        )
    }
}

/// Action evaluation result
#[derive(Clone, Debug)]
#[pyclass]
pub struct ActionScore {
    #[pyo3(get)]
    pub action: String,
    #[pyo3(get)]
    pub score: f32,
    #[pyo3(get)]
    pub predicted_outcome: String,
}

#[pymethods]
impl ActionScore {
    pub fn __repr__(&self) -> String {
        format!("ActionScore({}: {:.3})", self.action, self.score)
    }
}

/// Pollinator Configuration
#[derive(Clone, Debug)]
#[pyclass]
pub struct PollinatorConfig {
    #[pyo3(get, set)]
    pub recency_window: usize,
    #[pyo3(get, set)]
    pub sigmoid_temperature: f32,
    #[pyo3(get, set)]
    pub surprise_broadcast_weight: f32,
}

#[pymethods]
impl PollinatorConfig {
    #[new]
    #[pyo3(signature = (recency_window = 15, sigmoid_temperature = 1.0, surprise_broadcast_weight = 0.6))]
    pub fn new(recency_window: usize, sigmoid_temperature: f32, surprise_broadcast_weight: f32) -> Self {
        PollinatorConfig {
            recency_window,
            sigmoid_temperature,
            surprise_broadcast_weight,
        }
    }
}

/// Promoter Configuration
#[derive(Clone, Debug)]
#[pyclass]
pub struct PromoterConfig {
    #[pyo3(get, set)]
    pub density_threshold: usize,
    #[pyo3(get, set)]
    pub promotion_context: String,
}

#[pymethods]
impl PromoterConfig {
    #[new]
    #[pyo3(signature = (density_threshold = 8, promotion_context = "".to_string()))]
    pub fn new(density_threshold: usize, promotion_context: String) -> Self {
        PromoterConfig {
            density_threshold,
            promotion_context,
        }
    }
}
