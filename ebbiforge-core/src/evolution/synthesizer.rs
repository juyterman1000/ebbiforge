//! Tool Synthesizer
//!
//! Generates code for new tools based on agent requirements.

use super::GeneratedTool;
use pyo3::prelude::*;
use tracing::info;

/// Synthesizes new tools from descriptions
#[pyclass]
pub struct ToolSynthesizer {
    model_name: String,
}

#[pymethods]
impl ToolSynthesizer {
    #[new]
    #[pyo3(signature = (model_name = "gpt-4-turbo"))]
    pub fn new(model_name: &str) -> Self {
        info!("🧬 [Synthesizer] Initialized with backend: {}", model_name);
        ToolSynthesizer {
            model_name: model_name.to_string(),
        }
    }

    /// Request tool synthesis via the configured LLM backend.
    /// Returns an error if no LLM backend is active.
    pub fn synthesize(
        &self,
        name: String,
        _description: String,
        _implementation_hint: String,
    ) -> PyResult<GeneratedTool> {
        info!("🧬 [Synthesizer] Requesting '{}' generation via backend: {}", name, self.model_name);
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            format!("Tool synthesis requires an active LLM backend ('{}') and is disabled in this runtime.", self.model_name)
        ))
    }
}
