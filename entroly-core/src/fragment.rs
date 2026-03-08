/// Context Fragment — the atomic unit of managed context.
///
/// Mirrors Ebbiforge's Episode struct but optimized for context window
/// management rather than memory storage.
///
/// Scoring follows the ContextScorer pattern from ebbiforge-core/src/memory/lsh.rs:
///   composite = w_recency * recency + w_frequency * frequency
///             + w_semantic * semantic + w_entropy * entropy

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// A single piece of context (code snippet, file content, tool result, etc.)
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextFragment {
    #[pyo3(get, set)]
    pub fragment_id: String,
    #[pyo3(get, set)]
    pub content: String,
    #[pyo3(get, set)]
    pub token_count: u32,
    #[pyo3(get, set)]
    pub source: String,

    // Scoring components (all [0.0, 1.0])
    #[pyo3(get, set)]
    pub recency_score: f64,
    #[pyo3(get, set)]
    pub frequency_score: f64,
    #[pyo3(get, set)]
    pub semantic_score: f64,
    #[pyo3(get, set)]
    pub entropy_score: f64,

    // Metadata
    #[pyo3(get, set)]
    pub turn_created: u32,
    #[pyo3(get, set)]
    pub turn_last_accessed: u32,
    #[pyo3(get, set)]
    pub access_count: u32,
    #[pyo3(get, set)]
    pub is_pinned: bool,
    #[pyo3(get, set)]
    pub simhash: u64,

    /// Optional dense embedding vector (384-dim when embeddings feature is enabled).
    /// Used for cosine similarity in hybrid retrieval.
    #[serde(default)]
    pub embedding: Vec<f32>,

    // Hierarchical fragmentation: optional skeleton variant
    #[pyo3(get, set)]
    #[serde(default)]
    pub skeleton_content: Option<String>,
    #[pyo3(get, set)]
    #[serde(default)]
    pub skeleton_token_count: Option<u32>,

    /// Salience — controls per-fragment decay rate (ported from ebbiforge Episode.salience).
    ///
    /// Acts as a multiplier on the global decay_half_life:
    ///   effective_half_life = decay_half_life × salience
    ///
    /// Default 1.0 = standard decay rate.
    /// Higher = decays slower (fragment persists longer).
    /// Updated by:
    ///   - Recall reinforcement: ×1.2 each time optimize() selects this fragment
    ///   - Criticality boost: critical files start with salience > 1.0
    /// Capped at 5.0 to prevent unbounded growth.
    #[pyo3(get, set)]
    #[serde(default = "default_salience")]
    pub salience: f64,
}

fn default_salience() -> f64 { 1.0 }

#[pymethods]
impl ContextFragment {
    #[new]
    #[pyo3(signature = (fragment_id, content, token_count=0, source="".to_string()))]
    pub fn new(fragment_id: String, content: String, token_count: u32, source: String) -> Self {
        let tc = if token_count == 0 {
            (content.len() / 4).max(1) as u32
        } else {
            token_count
        };
        ContextFragment {
            fragment_id,
            content,
            token_count: tc,
            source,
            recency_score: 1.0,
            frequency_score: 0.0,
            semantic_score: 0.0,
            entropy_score: 0.5,
            turn_created: 0,
            turn_last_accessed: 0,
            access_count: 0,
            is_pinned: false,
            simhash: 0,
            embedding: Vec::new(),
            skeleton_content: None,
            skeleton_token_count: None,
            salience: 1.0,
        }
    }
}

/// Compute composite relevance score for a fragment.
///
/// Direct port of ebbiforge-core ContextScorer::score() but
/// with entropy replacing emotion as the fourth dimension.
///
/// `feedback_multiplier` comes from FeedbackTracker::learned_value():
///   > 1.0 = historically useful fragment (boosted)
///   < 1.0 = historically unhelpful fragment (suppressed)
///   = 1.0 = no feedback data (neutral)
#[inline]
pub fn compute_relevance(
    frag: &ContextFragment,
    w_recency: f64,
    w_frequency: f64,
    w_semantic: f64,
    w_entropy: f64,
    feedback_multiplier: f64,
) -> f64 {
    let total = w_recency + w_frequency + w_semantic + w_entropy;
    if total == 0.0 {
        return 0.0;
    }

    let base = (w_recency * frag.recency_score
        + w_frequency * frag.frequency_score
        + w_semantic * frag.semantic_score
        + w_entropy * frag.entropy_score)
        / total;

    (base * feedback_multiplier).min(1.0)
}

/// Apply Ebbinghaus forgetting curve decay to all fragments.
///
///   recency(t) = exp(-λ · Δt)
///   where λ = ln(2) / (half_life × salience)
///
/// Per-fragment salience scales the effective half-life: high-salience
/// fragments (frequently selected, critical files) decay slower.
/// Same math as ebbiforge-core HippocampusEngine.
pub fn apply_ebbinghaus_decay(
    fragments: &mut [ContextFragment],
    current_turn: u32,
    half_life: u32,
) {
    let base_half_life = half_life.max(1) as f64;

    for frag in fragments.iter_mut() {
        let effective_half_life = base_half_life * frag.salience.max(0.1);
        let decay_rate = (2.0_f64).ln() / effective_half_life;
        let dt = current_turn.saturating_sub(frag.turn_last_accessed) as f64;
        frag.recency_score = (-decay_rate * dt).exp();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ebbinghaus_half_life() {
        let mut frags = vec![ContextFragment::new(
            "x".into(), "test".into(), 10, "".into(),
        )];
        frags[0].turn_last_accessed = 0;

        apply_ebbinghaus_decay(&mut frags, 15, 15);

        // At exactly one half-life, recency should be ~0.5
        assert!((frags[0].recency_score - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_salience_slows_decay() {
        // Two fragments, same age, different salience
        let mut normal = ContextFragment::new("a".into(), "test".into(), 10, "".into());
        normal.turn_last_accessed = 0;
        normal.salience = 1.0;

        let mut critical = ContextFragment::new("b".into(), "test".into(), 10, "".into());
        critical.turn_last_accessed = 0;
        critical.salience = 3.0; // Critical file: 3× slower decay

        let mut frags = vec![normal, critical];
        apply_ebbinghaus_decay(&mut frags, 15, 15);

        // Normal decays to ~0.5 at one half-life
        assert!((frags[0].recency_score - 0.5).abs() < 0.01,
            "Normal salience should give ~0.5 at half-life, got {}", frags[0].recency_score);
        // Critical decays much slower (effective half-life = 45 turns)
        assert!(frags[1].recency_score > 0.75,
            "High salience should slow decay, got {}", frags[1].recency_score);
        assert!(frags[1].recency_score > frags[0].recency_score,
            "Critical fragment should retain more than normal");
    }

    #[test]
    fn test_salience_default_backward_compat() {
        let frag = ContextFragment::new("x".into(), "test".into(), 10, "".into());
        assert!((frag.salience - 1.0).abs() < 0.001, "Default salience should be 1.0");
    }

    #[test]
    fn test_relevance_scoring() {
        let mut frag = ContextFragment::new("a".into(), "test".into(), 10, "".into());
        frag.recency_score = 1.0;
        frag.frequency_score = 0.5;
        frag.semantic_score = 0.8;
        frag.entropy_score = 0.9;

        let score = compute_relevance(&frag, 0.30, 0.25, 0.25, 0.20, 1.0);
        assert!(score > 0.0 && score <= 1.0);

        // With positive feedback, score should be boosted
        let boosted = compute_relevance(&frag, 0.30, 0.25, 0.25, 0.20, 1.5);
        assert!(boosted > score);

        // With negative feedback, score should be suppressed
        let suppressed = compute_relevance(&frag, 0.30, 0.25, 0.25, 0.20, 0.6);
        assert!(suppressed < score);
    }
}
