use crate::TrajectoryPoint;
use parking_lot::RwLock;
use pyo3::prelude::*;
use std::sync::Arc;
use tracing::info;

/// Shared state passed through the Ebbiforge middleware pipeline.
#[pyclass]
#[derive(Clone, Default)]
pub struct EbbiforgeContext {
    #[pyo3(get, set)]
    pub task_id: String,
    #[pyo3(get, set)]
    pub prompt: String,
    #[pyo3(get, set)]
    pub should_stop: bool,
    #[pyo3(get, set)]
    pub stop_reason: Option<String>,
    /// Final answer from finish() tool - indicates successful completion
    #[pyo3(get, set)]
    pub final_answer: Option<String>,
    /// Trajectory stored as JSON for Python interop
    trajectory: Arc<RwLock<Vec<TrajectoryPoint>>>,
    /// Metadata as JSON string
    #[pyo3(get, set)]
    pub metadata_json: String,
}

#[pymethods]
impl EbbiforgeContext {
    #[new]
    pub fn new(task_id: String, prompt: String) -> Self {
        EbbiforgeContext {
            task_id,
            prompt,
            should_stop: false,
            stop_reason: None,
            final_answer: None,
            trajectory: Arc::new(RwLock::new(Vec::new())),
            metadata_json: "{}".to_string(),
        }
    }

    pub fn add_trajectory_point(&self, point: TrajectoryPoint) {
        let mut traj = self.trajectory.write();
        traj.push(point);
    }

    pub fn get_trajectory_json(&self) -> String {
        let traj = self.trajectory.read();
        serde_json::to_string(&*traj).unwrap_or("[]".to_string())
    }

    pub fn trajectory_len(&self) -> usize {
        let traj = self.trajectory.read();
        traj.len()
    }
}

/// A "Plugin" that hooks into the agent's lifecycle.
/// Simplified sync interface for easier PyO3 compatibility.
pub trait Middleware: Send + Sync {
    fn name(&self) -> &str;

    /// Ran BEFORE the agent executes a step.
    fn before_step(&self, _ctx: &mut EbbiforgeContext) -> Result<(), String> {
        Ok(())
    }

    /// Ran AFTER the agent executes a step.
    fn after_step(&self, _ctx: &mut EbbiforgeContext) -> Result<(), String> {
        Ok(())
    }

    /// Ran when an error occurs during execution.
    fn on_error(&self, _ctx: &mut EbbiforgeContext, _error: &str) -> Result<(), String> {
        Ok(())
    }
}

/// Container for middleware instances
pub struct MiddlewarePipeline {
    middlewares: Vec<Box<dyn Middleware>>,
}

impl MiddlewarePipeline {
    pub fn new() -> Self {
        MiddlewarePipeline {
            middlewares: Vec::new(),
        }
    }

    pub fn add(&mut self, middleware: Box<dyn Middleware>) {
        info!("🔌 [Pipeline] Registered middleware: {}", middleware.name());
        self.middlewares.push(middleware);
    }

    pub fn run_before(&self, ctx: &mut EbbiforgeContext) -> Result<(), String> {
        for mw in &self.middlewares {
            mw.before_step(ctx)?;
            if ctx.should_stop {
                return Ok(());
            }
        }
        Ok(())
    }

    pub fn run_after(&self, ctx: &mut EbbiforgeContext) -> Result<(), String> {
        for mw in &self.middlewares {
            mw.after_step(ctx)?;
        }
        Ok(())
    }

    pub fn run_error(&self, ctx: &mut EbbiforgeContext, error: &str) -> Result<(), String> {
        for mw in &self.middlewares {
            mw.on_error(ctx, error)?;
        }
        Ok(())
    }
}
