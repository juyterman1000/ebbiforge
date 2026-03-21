//! Implements Intrinsic Metacognition (State-of-the-art research).
//! Agents reflect on failures/successes to update knowledge.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::info;

/// Reflection insight
#[derive(Clone, Debug, Serialize, Deserialize)]
#[pyclass]
pub struct Insight {
    #[pyo3(get)]
    pub topic: String,
    #[pyo3(get)]
    pub observation: String,
    #[pyo3(get)]
    pub actionable_adjustment: String,
}

#[pymethods]
impl Insight {
    pub fn __repr__(&self) -> String {
        format!("Insight[{}]: {}", self.topic, self.actionable_adjustment)
    }
}

/// Metacognitive Engine
#[pyclass]
pub struct MetaCognition {
    knowledge_base: Vec<Insight>,
}

#[pymethods]
impl MetaCognition {
    #[new]
    pub fn new() -> Self {
        info!("[MetaCognition] Initialized intrinsic reflection engine");
        MetaCognition {
            knowledge_base: Vec::new(),
        }
    }

    /// Reflect on a task outcome
    /// In a real system, this uses an LLM to analyze the trace.
    /// Here we simulate heuristic reflection.
    pub fn reflect(&mut self, task_goal: String, success: bool, trace_summary: String) -> Insight {
        let insight = if success {
            Insight {
                topic: "Success Pattern".to_string(),
                observation: format!(
                    "Succeeded at '{}' using trace: {}",
                    task_goal, trace_summary
                ),
                actionable_adjustment: "Reinforce this strategy for similar tasks.".to_string(),
            }
        } else {
            // Simulate failure analysis
            let reason = if trace_summary.contains("ToolMissing") {
                "Synthesize a new tool"
            } else if trace_summary.contains("Timeout") {
                "Perform simpler sub-steps"
            } else {
                "Adjust reasoning prompt"
            };

            Insight {
                topic: "Failure Correction".to_string(),
                observation: format!("Failed at '{}'. Trace indicates issue.", task_goal),
                actionable_adjustment: format!("Strategy Update: {}.", reason),
            }
        };

        self.knowledge_base.push(insight.clone());
        info!("🤔 [Reflection] Generated: {:?}", insight);
        insight
    }

    /// Get all accumulated insights
    pub fn get_insights(&self) -> Vec<Insight> {
        self.knowledge_base.clone()
    }
}
