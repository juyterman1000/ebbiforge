//! Ebbiforge Core v4.0.0 - High-Performance AI Agent Runtime
//!
//! This library provides the core infrastructure for building and running
//! autonomous AI agents with a focus on safety, memory efficiency (zero-copy),
//! and production reliability.
// All modules re-exported for Python bindings

pub mod compliance;
pub mod core;
pub mod evolution;
pub mod intel;
pub mod security;
pub mod swarm;
pub mod utils;
pub mod worldmodel;

use parking_lot::RwLock;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// Re-export key types for internal use
pub use core::agent::{Agent, AgentRegistry};
pub use core::config::EbbiforgeConfig;
pub use core::middleware::{EbbiforgeContext, Middleware, MiddlewarePipeline};
pub use core::runner::{AgentGraph, AgentGraphPy};
pub use intel::safety::PredictiveSafetyShield;
pub use swarm::SwarmConfig;

/// A single snapshot of an agent's execution path.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[pyclass]
pub struct TrajectoryPoint {
    /// The step number in the execution sequence
    #[pyo3(get, set)]
    pub step: u32,
    /// The action taken by the agent (e.g., "Search", "CallTool")
    #[pyo3(get, set)]
    pub action: String,
    /// The generative thought or reasoning behind the action
    #[pyo3(get, set)]
    pub thought: String,
    /// Timestamp in milliseconds (used by safety shield pattern matching)
    #[pyo3(get, set)]
    #[serde(default)]
    pub timestamp_ms: u64,
}

impl TrajectoryPoint {
    /// Internal Rust constructor (timestamp defaults to 0)
    pub fn new(step: u32, action: String, thought: String) -> Self {
        TrajectoryPoint {
            step,
            action,
            thought,
            timestamp_ms: 0,
        }
    }
}

#[pymethods]
impl TrajectoryPoint {
    #[new]
    #[pyo3(signature = (step, action, thought, timestamp_ms=0))]
    pub fn py_new(step: u32, action: String, thought: String, timestamp_ms: u64) -> Self {
        TrajectoryPoint {
            step,
            action,
            thought,
            timestamp_ms,
        }
    }
}

/// A thread-safe, zero-copy history buffer for managing agent trajectories.
///
/// `HistoryBuffer` uses `Arc<RwLock>` to allow high-concurrency access and
/// O(1) shallow forking, enabling thousands of parallel agent simulations.
#[pyclass]
#[derive(Clone, Debug)]
pub struct HistoryBuffer {
    inner: Arc<RwLock<Vec<TrajectoryPoint>>>,
}

#[pymethods]
impl HistoryBuffer {
    #[new]
    pub fn new() -> Self {
        HistoryBuffer {
            inner: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn add(&self, item: TrajectoryPoint) {
        let mut data = self.inner.write();
        data.push(item);
    }

    pub fn len(&self) -> usize {
        let data = self.inner.read();
        data.len()
    }

    /// Creates a shallow copy (Zero-Copy fork)
    pub fn fork(&self) -> Self {
        HistoryBuffer {
            inner: self.inner.clone(),
        }
    }

    pub fn to_json(&self) -> String {
        let data = self.inner.read();
        serde_json::to_string(&*data).unwrap_or("[]".to_string())
    }

    pub fn last(&self) -> Option<TrajectoryPoint> {
        let data = self.inner.read();
        data.last().cloned()
    }

    pub fn last_action(&self) -> Option<String> {
        let data = self.inner.read();
        data.last().map(|p| p.action.clone())
    }

    pub fn get_raw(&self) -> Vec<TrajectoryPoint> {
        let data = self.inner.read();
        data.clone()
    }

    /// Merge another buffer's last item into this one.
    /// Used for aggregating results from parallel branches.
    pub fn merge(&self, other: &HistoryBuffer) {
        if let Some(last) = other.last() {
            self.add(last);
        }
    }
}

/// Initialize tracing for the library.
#[pyfunction]
pub fn setup_logging(level: Option<String>) {
    let filter = level.unwrap_or_else(|| "info".to_string());
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

/// Python module initialization
#[pymodule]
fn ebbiforge_core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core types
    m.add_class::<TrajectoryPoint>()?;
    m.add_class::<HistoryBuffer>()?;

    // Configuration
    m.add_class::<core::config::EbbiforgeConfig>()?;

    // Agents
    m.add_class::<core::agent::Agent>()?;
    m.add_class::<core::agent::AgentRegistry>()?;
    m.add_class::<core::runner::AgentGraphPy>()?;

    // Workflows
    m.add_class::<core::workflow::SequentialAgent>()?;
    m.add_class::<core::workflow::ParallelAgent>()?;
    m.add_class::<core::workflow::LoopAgent>()?;

    // Middleware
    m.add_class::<core::middleware::EbbiforgeContext>()?;
    m.add_class::<intel::safety::PredictiveSafetyShield>()?;

    // Swarm
    m.add_class::<swarm::SwarmConfig>()?;
    m.add_class::<swarm::TensorSwarm>()?;
    m.add_class::<intel::pruning::AdaptivePruner>()?;
    m.add_class::<intel::reviewer::CodeQualityGuard>()?;

    // Intelligence
    m.add_class::<intel::introspection::IntrospectionEngine>()?;

    // Utilities
    m.add_class::<intel::pruning::ContextFeatures>()?;
    m.add_class::<intel::pruning::ContextFragment>()?;
    m.add_class::<intel::pollination::FailureContext>()?;
    m.add_class::<intel::pollination::ExperiencePack>()?;
    m.add_class::<intel::pollination::CrossPollination>()?;
    m.add_class::<utils::benchmark::AgentBenchmark>()?;

    // Shared Memory (Embeddings)
    m.add_class::<core::shared_memory::SharedMemoryStore>()?;

    // Storage Backends
    m.add_class::<core::storage::StorageConfig>()?;
    m.add_class::<core::storage::SearchResult>()?;
    m.add_class::<core::storage::dragonfly::DragonflyStore>()?;
    m.add_class::<core::storage::remote_vector::RemoteVectorStore>()?;
    // Compliance (Regulated AI)
    m.add_class::<compliance::ComplianceEngine>()?;
    m.add_class::<compliance::ComplianceResult>()?;
    m.add_class::<compliance::pii::PIIRedactor>()?;
    m.add_class::<compliance::pii::PIIMatch>()?;
    m.add_class::<compliance::audit::AuditEvent>()?;
    m.add_class::<compliance::policy::Policy>()?;
    m.add_class::<compliance::trace::TraceStep>()?;

    // Enterprise OWASP (Rate Limiting, Escalation, Sanitization)
    m.add_class::<compliance::ratelimit::RateLimiter>()?;
    m.add_class::<compliance::ratelimit::RateLimitConfig>()?;
    m.add_class::<compliance::ratelimit::RateLimitResult>()?;
    m.add_class::<compliance::escalation::EscalationFlow>()?;
    m.add_class::<compliance::escalation::PendingAction>()?;
    m.add_class::<compliance::escalation::EscalationResult>()?;
    m.add_class::<compliance::sanitizer::InputSanitizer>()?;
    m.add_class::<compliance::sanitizer::SanitizeResult>()?;

    // Security (Secure Multi-Agent Trust)
    m.add_class::<core::security::AgentIdentity>()?;
    m.add_class::<core::security::TrustStore>()?;

    // Latent World Model (Predictive Planning)
    m.add_class::<worldmodel::LatentState>()?;
    m.add_class::<worldmodel::WorldModelConfig>()?;
    m.add_class::<worldmodel::Prediction>()?;
    m.add_class::<worldmodel::ActionScore>()?;
    m.add_class::<worldmodel::LatentEncoder>()?;
    m.add_class::<worldmodel::AutoregressivePredictor>()?;
    m.add_class::<worldmodel::PlanningEngine>()?;
    m.add_class::<worldmodel::MemoryConsolidator>()?;
    m.add_class::<worldmodel::consolidator::ConsolidatedMemory>()?;
    m.add_class::<worldmodel::PollinatorConfig>()?;
    m.add_class::<worldmodel::PromoterConfig>()?;

    // v9 Geometric & Diffusion
    m.add_class::<worldmodel::GeometricEncoder>()?;
    m.add_class::<worldmodel::DiffusionPredictor>()?;

    // Trajectory Training (Predictive Learning)
    m.add_class::<worldmodel::TrajectoryBuffer>()?;
    m.add_class::<worldmodel::dynamics::TrainStats>()?;

    // v10 Self-Evolution (MetaAgent)
    m.add_class::<evolution::EvolutionConfig>()?;
    m.add_class::<evolution::GeneratedTool>()?;
    m.add_class::<evolution::ToolSynthesizer>()?;
    m.add_class::<evolution::SafetySandbox>()?;
    m.add_class::<evolution::DynamicRegistry>()?;

    // v11 Darwinian & Metacognitive
    m.add_class::<evolution::AgentGenome>()?;
    m.add_class::<evolution::PopulationEngine>()?;
    m.add_class::<evolution::MetaCognition>()?;
    m.add_class::<evolution::Insight>()?;
    m.add_class::<evolution::CuriosityModule>()?;

    // v12 Swarm Engine — Nervous System
    m.add_class::<swarm::SwarmConfig>()?;
    m.add_class::<swarm::TensorSwarm>()?;
    m.add_class::<swarm::tensor_engine::TensorSwarm>()?;
    m.add_class::<swarm::promoter::PromotionLogic>()?;
    m.add_class::<swarm::pollination::PollinatorState>()?;
    m.add_class::<swarm::ProductionTensorSwarm>()?;
    m.add_class::<swarm::DormantAgent>()?;
    m.add_class::<swarm::SimplifiedPool>()?;

    // Nervous System — Signal Ingestion & Spatial Awareness
    m.add_class::<swarm::watcher::SwarmWatcher>()?;
    m.add_class::<swarm::watcher::Signal>()?;
    m.add_class::<swarm::spatial::GridMap>()?;
    m.add_class::<swarm::py_api::PySwarmEngine>()?;

    // Security (Encryption)
    m.add_class::<security::aes::SecureVault>()?;

    Ok(())
}
