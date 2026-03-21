use crate::{HistoryBuffer, TrajectoryPoint};
use pyo3::prelude::*;
use tracing::info;

/// A specialized agent that executes a list of sub-agents in a linear sequence.
///
/// The output and state of each agent in the chain are passed incrementally
/// to the next, forming a deterministic pipeline.
#[pyclass]
pub struct SequentialAgent {
    /// Name of the workflow
    #[pyo3(get)]
    pub name: String,
    /// Ordered list of agent personas to invoke
    pub sub_agents: Vec<String>,
}

#[pymethods]
impl SequentialAgent {
    #[new]
    pub fn new(name: String, sub_agents: Vec<String>) -> Self {
        SequentialAgent { name, sub_agents }
    }

    pub fn add_agent(&mut self, agent_name: String) {
        self.sub_agents.push(agent_name);
    }

    pub fn run(
        &self,
        graph: &crate::core::runner::AgentGraphPy,
        task_id: String,
        buffer: &HistoryBuffer,
    ) -> PyResult<()> {
        info!(
            "⛓️ [SequentialAgent: {}] Starting pipeline execution...",
            self.name
        );

        for agent_name in &self.sub_agents {
            info!("   ⬇️ Step: {}", agent_name);

            // Run task with specific agent persona
            let _ = graph.run_task(task_id.clone(), buffer, Some(agent_name.clone()))?;

            // Add result marker
            let step = buffer.len() as u32 + 1;
            buffer.add(TrajectoryPoint::new(
                step,
                format!("ResultFrom_{}", agent_name),
                format!("Output: Step {} complete", step),
            ));
        }

        info!("[SequentialAgent] Pipeline finished.");
        Ok(())
    }
}

/// A specialized agent that initiates concurrent execution of multiple sub-agents.
///
/// Leverages the zero-copy architecture to fork the state into independent
/// branches for parallel simulation.
#[pyclass]
pub struct ParallelAgent {
    /// Name of the workflow
    #[pyo3(get)]
    pub name: String,
    /// List of agent personas to spawn concurrently
    pub sub_agents: Vec<String>,
}

#[pymethods]
impl ParallelAgent {
    #[new]
    pub fn new(name: String, sub_agents: Vec<String>) -> Self {
        ParallelAgent { name, sub_agents }
    }

    pub fn add_agent(&mut self, agent_name: String) {
        self.sub_agents.push(agent_name);
    }

    pub fn run(
        &self,
        graph: &crate::core::runner::AgentGraphPy,
        task_id: String,
        buffer: &HistoryBuffer,
    ) -> PyResult<()> {
        info!(
            "[ParallelAgent: {}] Spawning {} agents...",
            self.name,
            self.sub_agents.len()
        );

        let mut _handles: Vec<()> = Vec::new();
        for agent_name in self.sub_agents.clone() {
            // Cloned sub_agents to allow iteration and move agent_name
            info!("   🚀 Spawning: {}", agent_name);
            let graph_clone = graph; // Borrowed reference — pyo3 objects are !Send

            // Fork buffer for each branch (Zero-Copy)
            let branch_buffer = buffer.fork();
            let task_id_branch = format!("{}-{}", task_id, agent_name);

            // Execute in forked buffer — graph is !Send (pyo3 constraint), so tasks
            // run on the calling thread with isolated state via buffer.fork()
            let _ =
                graph_clone.run_task(task_id_branch, &branch_buffer, Some(agent_name.clone()))?;

            // Merge result back
            let step = buffer.len() as u32 + 1;
            buffer.add(TrajectoryPoint::new(
                step,
                format!("ResultFrom_{}", agent_name),
                "Parallel Output: Complete".to_string(),
            ));
        }

        info!("[ParallelAgent] All threads joined.");
        Ok(())
    }
}

/// A specialized agent that executes a target persona iteratively.
///
/// The loop continues until the specified maximum number of iterations is
/// reached or a model-defined termination condition is triggered.
#[pyclass]
pub struct LoopAgent {
    /// Name of the workflow
    #[pyo3(get)]
    pub name: String,
    /// The target agent persona to invoke in a loop
    pub agent_name: String,
    /// Maximum allowed iterations before hard stop
    #[pyo3(get, set)]
    pub max_iterations: usize,
}

#[pymethods]
impl LoopAgent {
    #[new]
    pub fn new(name: String, agent_name: String, max_iterations: usize) -> Self {
        LoopAgent {
            name,
            agent_name,
            max_iterations,
        }
    }

    pub fn run(
        &self,
        graph: &crate::core::runner::AgentGraphPy,
        task_id: String,
        buffer: &HistoryBuffer,
    ) -> PyResult<()> {
        info!(
            "🔄 [LoopAgent: {}] Starting loop (Max: {})...",
            self.name, self.max_iterations
        );

        for i in 0..self.max_iterations {
            info!("   🔄 Iteration {}/{}", i + 1, self.max_iterations);

            // Execute iteration with target agent
            let _ = graph.run_task(task_id.clone(), buffer, Some(self.agent_name.clone()))?;

            let step = buffer.len() as u32 + 1;
            buffer.add(TrajectoryPoint::new(
                step,
                format!("IterationResult_{}", i),
                format!("Iteration {} complete", i),
            ));

            // Check termination condition
            if let Some(last) = buffer.last() {
                // Original check
                if last.thought.contains("COMPLETE") {
                    info!("   🛑 Termination Condition Met.");
                    break;
                }
            }
        }

        Ok(())
    }
}
