//! Latent Encoder for compressing context to fixed-size vectors
//!
//! Converts trajectory context AND goals into compact latent representations.
//! Updated for context-aware language conditioning with native ONNX semantic embeddings.

use super::{LatentState, WorldModelConfig};
use crate::TrajectoryPoint;
use pyo3::prelude::*;
use tracing::info;
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};

use parking_lot::RwLock;

/// Latent encoder for trajectory compression
#[pyclass]
pub struct LatentEncoder {
    config: WorldModelConfig,
    model: RwLock<TextEmbedding>,
}

#[pymethods]
impl LatentEncoder {
    #[new]
    #[pyo3(signature = (config = None))]
    pub fn new(config: Option<WorldModelConfig>) -> pyo3::PyResult<Self> {
        let cfg = config.unwrap_or_default();
        info!("[Encoder] Initializing fastembed ONNX runtime natively...");

        let model = TextEmbedding::try_new(InitOptions::new(EmbeddingModel::BGEBaseENV15))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Fastembed error: {}", e)))?;

        info!("[Encoder] Initialized (dim={})", cfg.latent_dim);
        Ok(LatentEncoder { config: cfg, model: RwLock::new(model) })
    }

    /// Encode a trajectory + goal context into a latent state
    pub fn encode_context(
        &self,
        trajectory_json: String,
        goal: String,
        agent_id: String,
    ) -> LatentState {
        // Parse trajectory
        let trajectory: Vec<TrajectoryPoint> =
            serde_json::from_str(&trajectory_json).unwrap_or_default();

        let mut vector = vec![0.0f32; self.config.latent_dim];
        let mut texts = Vec::new();
        let mut weights = Vec::new();

        // 1. Encode Trajectory Context (Past)
        let window_size = self.config.context_window.min(trajectory.len());
        let recent: Vec<&TrajectoryPoint> = trajectory.iter().rev().take(window_size).collect();

        for (i, point) in recent.iter().enumerate() {
            let weight = 0.6 / (i as f32 + 1.0); // 60% weight to history
            texts.push(format!("Action: {} Thought: {}", point.action, point.thought));
            weights.push(weight);
        }

        // 2. Encode Goal Context (Future Intent)
        if !goal.is_empty() {
            texts.push(format!("Goal: {}", goal));
            weights.push(0.4); // 40% weight to goal
        }

        if texts.is_empty() {
             texts.push("empty".to_string());
             weights.push(1.0);
        }

        let mut model_lock = self.model.write();
        if let Ok(embeddings) = model_lock.embed(texts, None) {
             for (emb, weight) in embeddings.iter().zip(weights.iter()) {
                 for j in 0..self.config.latent_dim.min(emb.len()) {
                     vector[j] += emb[j] * weight;
                 }
             }
        }

        // Normalize
        self.normalize(&mut vector);

        LatentState::new(vector, agent_id, trajectory.len() as u32)
    }

    /// Encode just the trajectory (backward handling)
    pub fn encode(&self, trajectory_json: String, agent_id: String) -> LatentState {
        self.encode_context(trajectory_json, String::new(), agent_id)
    }

    /// Encode a natural language action into latent space
    pub fn encode_action(&self, action: String) -> Vec<f32> {
        let mut vector = vec![0.0f32; self.config.latent_dim];
        let texts = vec![if action.is_empty() { "empty".to_string() } else { action }];
        
        let mut model_lock = self.model.write();
        if let Ok(embeddings) = model_lock.embed(texts, None) {
             if let Some(emb) = embeddings.first() {
                 for i in 0..self.config.latent_dim.min(emb.len()) {
                     vector[i] = emb[i];
                 }
             }
        }

        self.normalize(&mut vector);
        vector
    }

    /// Decode latent state back to summary
    pub fn decode(&self, state: &LatentState) -> String {
        let magnitude: f32 =
            state.vector.iter().map(|x| x.abs()).sum::<f32>() / state.vector.len() as f32;
        format!(
            "LatentContext(agent={}, step={}, dim={}, activation={:.3})",
            state.agent_id,
            state.step,
            state.vector.len(),
            magnitude
        )
    }
}

impl LatentEncoder {
    fn normalize(&self, vector: &mut Vec<f32>) {
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in vector.iter_mut() {
                *v /= norm;
            }
        }
    }
}
