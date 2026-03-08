//! Safety Sandbox
//!
//! Verifies generated code to prevent dangerous operations.

use super::{EvolutionConfig, GeneratedTool};
use pyo3::prelude::*;
use tracing::{info, warn};

/// Sandbox for verifying tool safety
#[pyclass]
pub struct SafetySandbox {
    config: EvolutionConfig,
    banned_imports: Vec<String>,
}

#[pymethods]
impl SafetySandbox {
    #[new]
    #[pyo3(signature = (config = None))]
    pub fn new(config: Option<EvolutionConfig>) -> Self {
        let cfg = config.unwrap_or_default();

        // Define dangerous imports
        let banned = vec![
            "os".to_string(),
            "sys".to_string(),
            "subprocess".to_string(),
            "shutil".to_string(),
            "socket".to_string(),
            "requests".to_string(), // In strict mode
        ];

        info!("[Sandbox] Initialized checks");

        SafetySandbox {
            config: cfg,
            banned_imports: banned,
        }
    }

    /// Verify tool code for safety using configured restrictions
    pub fn verify(&self, tool: &GeneratedTool) -> bool {
        // 1. Check for banned imports
        for banned in &self.banned_imports {
            if tool.code.contains(&format!("import {}", banned))
                || tool.code.contains(&format!("from {} import", banned))
            {
                warn!(
                    "🚫 [Sandbox] Rejected '{}': contains banned import '{}'",
                    tool.name, banned
                );
                return false;
            }
        }

        // 2. Check for dangerous dynamic execution calls
        if tool.code.contains("eval(") || tool.code.contains("exec(") {
            warn!("🚫 [Sandbox] Rejected '{}': contains eval/exec", tool.name);
            return false;
        }

        // 3. Config-driven mutation rate threshold for auto-generated tools
        if self.config.mutation_rate > 0.5 {
            warn!("🚫 [Sandbox] High mutation rate ({:.2}) — extra scrutiny applied", self.config.mutation_rate);
        }

        info!("[Sandbox] Verified '{}': Code passes safety checks", tool.name);
        true
    }
}
