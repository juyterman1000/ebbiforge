
use crate::{HistoryBuffer, TrajectoryPoint};
use pyo3::prelude::*;
use std::sync::Arc;
use tokio::runtime::Runtime;
use tracing::info;

pub struct AgentGraph {
    runtime: Arc<Runtime>,
}

impl AgentGraph {
    pub fn new() -> Self {
        AgentGraph {
            runtime: Arc::new(Runtime::new().unwrap_or_else(|_| std::process::abort())),
        }
    }

    /// Thread-safe execution using the Shared Reference buffer.
    pub fn run_task(&self, task_id: String, buffer: &HistoryBuffer) -> PyResult<()> {
        // Enter the tokio runtime context for async compatibility
        let _guard = self.runtime.enter();

        info!("▶️ [RustCore] Starting Task: {}", task_id);

        let current_len = buffer.len();
        info!("   History Depth: {} (Zero-Copy Ref)", current_len);

        // Record task execution in the trajectory buffer
        let new_point = TrajectoryPoint {
            step: (current_len + 1) as u32,
            action: "ComputedInRust".to_string(),
            thought: format!("Processed {} items in O(1) memory cost.", current_len),
        };

        buffer.add(new_point);
        Ok(())
    }
}
