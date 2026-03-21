//! Curiosity Module (AgentEvolver)
//!
//! Generates self-directed goals to drive exploration.

use pyo3::prelude::*;
use rand::prelude::*;
use tracing::info;

/// Curiosity Generator
#[pyclass]
pub struct CuriosityModule {
    verified_skills: Vec<String>,
}

#[pymethods]
impl CuriosityModule {
    #[new]
    pub fn new() -> Self {
        info!("🔭 [Curiosity] Initialized exploration engine");
        CuriosityModule {
            verified_skills: Vec::new(),
        }
    }

    pub fn learn_skill(&mut self, skill: String) {
        if !self.verified_skills.contains(&skill) {
            self.verified_skills.push(skill);
        }
    }

    /// Suggest a new challenge task based on current skills
    pub fn propose_challenge(&self) -> String {
        let mut rng = thread_rng();

        if self.verified_skills.is_empty() {
            return "Explore basic system capabilities".to_string();
        }

        // Combine known skills into a novel task (Combinatorial Creativity)
        let default_skill = String::new();
        let skill = self
            .verified_skills
            .choose(&mut rng)
            .unwrap_or(&default_skill);

        let challenges = vec![
            format!("Stress test '{}' with 1000 iterations", skill),
            format!("Chain '{}' with file system operations", skill),
            format!("Optimize '{}' for speed", skill),
            format!("Find edge cases where '{}' fails", skill),
        ];

        let challenge = challenges
            .choose(&mut rng)
            .map(|s| s.clone())
            .unwrap_or_default();
        info!("[Curiosity] Proposed new challenge: {}", challenge);
        challenge
    }
}
