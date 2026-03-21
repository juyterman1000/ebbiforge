use crate::core::middleware::{EbbiforgeContext, Middleware};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::info;

/// Features for context scoring
#[derive(Clone, Debug, Serialize, Deserialize)]
#[pyclass]
pub struct ContextFeatures {
    #[pyo3(get, set)]
    pub recency: f64,
    #[pyo3(get, set)]
    pub relevance: f64,
    #[pyo3(get, set)]
    pub historical_success: f64,
    #[pyo3(get, set)]
    pub complexity: f64,
}

#[pymethods]
impl ContextFeatures {
    #[new]
    pub fn new(recency: f64, relevance: f64, historical_success: f64, complexity: f64) -> Self {
        ContextFeatures {
            recency,
            relevance,
            historical_success,
            complexity,
        }
    }
}

/// A fragment of context with features
#[derive(Clone, Debug, Serialize, Deserialize)]
#[pyclass]
pub struct ContextFragment {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub text: String,
    pub features: ContextFeatures,
}

#[pymethods]
impl ContextFragment {
    #[new]
    pub fn new(
        id: String,
        text: String,
        recency: f64,
        relevance: f64,
        historical_success: f64,
        complexity: f64,
    ) -> Self {
        ContextFragment {
            id,
            text,
            features: ContextFeatures::new(recency, relevance, historical_success, complexity),
        }
    }

    pub fn get_features(&self) -> ContextFeatures {
        self.features.clone()
    }
}

/// Policy weights for pruning
#[derive(Clone, Debug)]
pub struct PruningPolicy {
    pub recency_weight: f64,
    pub relevance_weight: f64,
    pub historical_success_weight: f64,
    pub complexity_weight: f64,
    pub learning_rate: f64,
}

impl Default for PruningPolicy {
    fn default() -> Self {
        PruningPolicy {
            recency_weight: 0.2,
            relevance_weight: 0.5,
            historical_success_weight: 0.5,
            complexity_weight: -0.1,
            learning_rate: 0.1,
        }
    }
}

/// Adaptive Pruner using RL-style weight updates
#[pyclass]
pub struct AdaptivePruner {
    policy: PruningPolicy,
}

#[pymethods]
impl AdaptivePruner {
    #[new]
    pub fn new() -> Self {
        AdaptivePruner {
            policy: PruningPolicy::default(),
        }
    }

    /// Score a fragment based on policy weights
    pub fn score_fragment(
        &self,
        recency: f64,
        relevance: f64,
        historical_success: f64,
        complexity: f64,
    ) -> f64 {
        (recency * self.policy.recency_weight)
            + (relevance * self.policy.relevance_weight)
            + (historical_success * self.policy.historical_success_weight)
            + (complexity * self.policy.complexity_weight)
    }

    /// Prune fragments to fit within target length
    pub fn prune(&self, fragments_json: String, target_length: usize) -> String {
        let fragments: Vec<ContextFragment> =
            serde_json::from_str(&fragments_json).unwrap_or_default();

        // Score and sort
        let mut scored: Vec<(ContextFragment, f64)> = fragments
            .into_iter()
            .map(|f| {
                let score = self.score_fragment(
                    f.features.recency,
                    f.features.relevance,
                    f.features.historical_success,
                    f.features.complexity,
                );
                (f, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select top fragments within limit
        let mut result = String::new();
        for (fragment, _) in scored {
            if result.len() + fragment.text.len() <= target_length {
                result.push_str(&fragment.text);
                result.push_str("\n---\n");
            }
        }

        result
    }

    /// Update policy based on feedback (simplified RL)
    pub fn update_policy(
        &mut self,
        feedback: f64,
        recency: f64,
        relevance: f64,
        historical_success: f64,
        complexity: f64,
    ) {
        let lr = self.policy.learning_rate;

        self.policy.recency_weight += lr * feedback * recency;
        self.policy.relevance_weight += lr * feedback * relevance;
        self.policy.historical_success_weight += lr * feedback * historical_success;
        self.policy.complexity_weight += lr * feedback * complexity;

        // Normalize weights to [-1, 1]
        self.policy.recency_weight = self.policy.recency_weight.clamp(-1.0, 1.0);
        self.policy.relevance_weight = self.policy.relevance_weight.clamp(-1.0, 1.0);
        self.policy.historical_success_weight =
            self.policy.historical_success_weight.clamp(-1.0, 1.0);
        self.policy.complexity_weight = self.policy.complexity_weight.clamp(-1.0, 1.0);

        self.policy.complexity_weight = self.policy.complexity_weight.clamp(-1.0, 1.0);

        info!(
            "[AdaptivePruner] Updated Policy: rec={:.2}, rel={:.2}, hist={:.2}, comp={:.2}",
            self.policy.recency_weight,
            self.policy.relevance_weight,
            self.policy.historical_success_weight,
            self.policy.complexity_weight
        );
    }
}

impl Middleware for AdaptivePruner {
    fn name(&self) -> &str {
        "AdaptivePruner"
    }

    fn before_step(&self, _ctx: &mut EbbiforgeContext) -> Result<(), String> {
        info!("✂️ [AdaptivePruner] Ready to optimize context.");
        Ok(())
    }

    fn after_step(&self, _ctx: &mut EbbiforgeContext) -> Result<(), String> {
        Ok(())
    }
}
