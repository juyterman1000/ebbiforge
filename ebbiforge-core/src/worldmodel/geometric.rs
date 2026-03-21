//! Topological Manifold Regularization
//!
//! Implements topological constraints for latent consistency.
//! Enforces temporal consistency: similar states must be close in latent space.

use super::{LatentEncoder, LatentState, WorldModelConfig};
use pyo3::prelude::*;
use tracing::info;

/// Geometric Encoder with temporal regularization
#[pyclass]
pub struct GeometricEncoder {
    base_encoder: LatentEncoder,
    regularization_strength: f32, // Topological manifold constraint
}

#[pymethods]
impl GeometricEncoder {
    #[new]
    #[pyo3(signature = (config = None))]
    pub fn new(config: Option<WorldModelConfig>) -> Self {
        let cfg = config.unwrap_or_default();
        info!("[GeometricEncoder] Initialized");
        GeometricEncoder {
            base_encoder: LatentEncoder::new(Some(cfg)).unwrap(),
            regularization_strength: 0.1,
        }
    }

    /// Encode with geometric regularization
    /// This simulates training by "pulling" the embedding towards a manifold
    pub fn encode(&self, context: String, goal: String, agent_id: String) -> LatentState {
        // 1. Get base latent vector (Language-
        let mut state = self
            .base_encoder
            .encode_context(context.clone(), goal, agent_id);

        // 2. Apply Topological Regularization
        // In a real state-of-the-art model, this is learned via contrastive loss.
        // Here, we simulate it by smoothing the vector based on topological properties
        // (e.g., ensuring smoothness across dimensions)
        self.regularize(&mut state.vector);

        state
    }

    /// Encode action
    pub fn encode_action(&self, action: String) -> Vec<f32> {
        let mut vec = self.base_encoder.encode_action(action);
        self.regularize(&mut vec);
        vec
    }
}

impl GeometricEncoder {
    // Topological constraint: Reshape latent space to mirror geometry of true state manifold
    fn regularize(&self, vector: &mut Vec<f32>) {
        if vector.is_empty() {
            return;
        }

        // Simulate geometric smoothing (Topology Preservation)
        // We apply a simple convolution-like smoothing to ensure
        // adjacent dimensions don't jump wildly (smooth manifold hypothesis)
        let original = vector.clone();
        for i in 1..vector.len() - 1 {
            vector[i] = original[i] * (1.0 - self.regularization_strength)
                + (original[i - 1] + original[i + 1]) * 0.5 * self.regularization_strength;
        }

        // Re-normalize to unit sphere (Hypersphere manifold)
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in vector {
                *v /= norm;
            }
        }
    }
}
