//! Evolution Layer for Self-Improving Agents
//!
//! Enables agents to:
//! 1. Synthesize new tools on the fly (MetaAgent)
//! 2. Execute tools safely in a sandbox (Top-tier safety)
//! 3. Register tools dynamically for immediate use

pub mod curiosity;
pub mod metacognition;
pub mod population;
pub mod registry;
pub mod sandbox;
pub mod synthesizer;

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

pub use curiosity::CuriosityModule;
pub use metacognition::{Insight, MetaCognition};
pub use population::{AgentGenome, PopulationEngine};
pub use registry::DynamicRegistry;
pub use sandbox::SafetySandbox;
pub use synthesizer::ToolSynthesizer;

/// Configuration for agent evolution
#[derive(Clone, Debug)]
#[pyclass]
pub struct EvolutionConfig {
    #[pyo3(get, set)]
    pub allow_synthesis: bool,
    #[pyo3(get, set)]
    pub safety_level: String, // "strict", "moderate", "lenient"
    #[pyo3(get, set)]
    pub max_tools: usize,
    #[pyo3(get, set)]
    pub population_size: usize,
    #[pyo3(get, set)]
    pub mutation_rate: f32,
}

#[pymethods]
impl EvolutionConfig {
    #[new]
    #[pyo3(signature = (allow_synthesis = true, safety_level = "strict", max_tools = 50, population_size = 5, mutation_rate = 0.1))]
    pub fn new(
        allow_synthesis: bool,
        safety_level: &str,
        max_tools: usize,
        population_size: usize,
        mutation_rate: f32,
    ) -> Self {
        EvolutionConfig {
            allow_synthesis,
            safety_level: safety_level.to_string(),
            max_tools,
            population_size,
            mutation_rate,
        }
    }
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self::new(true, "strict", 50, 5, 0.1)
    }
}

/// A synthetically generated tool
#[derive(Clone, Debug, Serialize, Deserialize)]
#[pyclass]
pub struct GeneratedTool {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub code: String,
    #[pyo3(get)]
    pub description: String,
    #[pyo3(get)]
    pub created_at: u64,
    #[pyo3(get, set)]
    pub verified: bool,
}

#[pymethods]
impl GeneratedTool {
    pub fn __repr__(&self) -> String {
        format!(
            "GeneratedTool(name='{}', verified={})",
            self.name, self.verified
        )
    }
}
