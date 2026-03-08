use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Safety configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
#[pyclass]
pub struct SafetyConfig {
    #[pyo3(get, set)]
    pub risk_threshold: f64,
    #[pyo3(get, set)]
    pub max_risk_history: usize,
}

/// Pruning configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
#[pyclass]
pub struct PruningConfig {
    #[pyo3(get, set)]
    pub target_length: usize,
    #[pyo3(get, set)]
    pub complexity_penalty: f64,
}

/// Introspection configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
#[pyclass]
pub struct IntrospectionConfig {
    #[pyo3(get, set)]
    pub drift_threshold: f64,
    #[pyo3(get, set)]
    pub loop_detection_window: usize,
}

/// Main hyperparameters for CogOps
#[derive(Clone, Debug, Serialize, Deserialize)]
#[pyclass]
pub struct EbbiforgeConfig {
    #[pyo3(get, set)]
    pub safety: SafetyConfig,
    #[pyo3(get, set)]
    pub pruning: PruningConfig,
    #[pyo3(get, set)]
    pub introspection: IntrospectionConfig,
    #[pyo3(get, set)]
    pub system_prompt: String,
}

#[pymethods]
impl EbbiforgeConfig {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for EbbiforgeConfig {
    fn default() -> Self {
        EbbiforgeConfig {
            safety: SafetyConfig {
                risk_threshold: 0.5,
                max_risk_history: 10,
            },
            pruning: PruningConfig {
                target_length: 100,
                complexity_penalty: -0.1,
            },
            introspection: IntrospectionConfig {
                drift_threshold: 0.3,
                loop_detection_window: 3,
            },
            system_prompt: "You are a research agent. Use the tools to find REAL information.\n\n\
                IMPORTANT RULES:\n\
                1. ALWAYS use web_search to find current data (stock prices, distances, etc.)\n\
                2. Use calculate for any math\n\
                3. Call finish(answer) when you have the final answer\n\
                4. DO NOT say 'I cannot access real-time data' - use the tools!".to_string(),
        }
    }
}
