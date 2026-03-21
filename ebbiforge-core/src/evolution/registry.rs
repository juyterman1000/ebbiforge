//! Dynamic Registry
//!
//! Manages hot-loaded tools available to the agent.

use super::GeneratedTool;
use parking_lot::RwLock;
use pyo3::prelude::*;
use std::collections::HashMap;
use tracing::info;

/// Registry for evolving agent capabilities
#[pyclass]
pub struct DynamicRegistry {
    tools: RwLock<HashMap<String, GeneratedTool>>,
}

#[pymethods]
impl DynamicRegistry {
    #[new]
    pub fn new() -> Self {
        DynamicRegistry {
            tools: RwLock::new(HashMap::new()),
        }
    }

    /// Register a new verified tool
    pub fn register(&self, tool: GeneratedTool) -> PyResult<()> {
        if !tool.verified {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Cannot register unverified tool: {}",
                tool.name
            )));
        }

        let mut map = self.tools.write();
        map.insert(tool.name.clone(), tool.clone());
        info!("📚 [Registry] Registered new capability: {}", tool.name);
        Ok(())
    }

    /// Get tool code by name
    pub fn get_tool_code(&self, name: String) -> Option<String> {
        let map = self.tools.read();
        map.get(&name).map(|t| t.code.clone())
    }

    /// List all available dynamic tools
    pub fn list_tools(&self) -> Vec<String> {
        let map = self.tools.read();
        map.keys().cloned().collect()
    }
}
