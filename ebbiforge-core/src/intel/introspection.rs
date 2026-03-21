use crate::core::runner::AgentGraphPy;
use crate::{HistoryBuffer, TrajectoryPoint};
use pyo3::prelude::*;
use tracing::{info, warn};

/// Implements a self-correction pattern (Academic research).
/// Wraps a Generator Agent and a Critic Agent in a self-correction loop.
#[pyclass]
pub struct IntrospectionEngine {
    #[pyo3(get, set)]
    pub max_attempts: usize,
}

#[pymethods]
impl IntrospectionEngine {
    #[new]
    pub fn new(max_attempts: usize) -> Self {
        IntrospectionEngine { max_attempts }
    }

    /// Runs the task with a self-correction loop.
    pub fn run_with_self_correction(
        &self,
        graph: &AgentGraphPy,
        task_id: String,
        buffer: &HistoryBuffer,
        generator_name: String,
        critic_name: String,
    ) -> PyResult<()> {
        info!(
            "[IntrospectionEngine] Starting self-correction Loop (Max {} Attempts)...",
            self.max_attempts
        );

        for attempt in 1..=self.max_attempts {
            info!("   🔄 Attempt {}/{}", attempt, self.max_attempts);

            // 1. GENERATE
            info!("   📝 Generator ({}) is thinking...", generator_name);
            let gen_task_id = format!("{}-Att{}", task_id, attempt);
            let _ = graph.run_task(gen_task_id, buffer, Some(generator_name.clone()))?;

            let step = buffer.len() as u32 + 1;
            buffer.add(TrajectoryPoint::new(
                step,
                "Generation".to_string(),
                format!("Attempt {} Output", attempt),
            ));

            // 2. CRITIQUE
            info!("   🧐 Critic ({}) is evaluating...", critic_name);

            // Add system prompt for critic
            let step = buffer.len() as u32 + 1;
            buffer.add(TrajectoryPoint::new(
                step,
                "System".to_string(),
                "Please evaluate the previous output. If correct/safe, say APPROVED.".to_string(),
            ));

            let crit_task_id = format!("{}-Crit{}", task_id, attempt);
            let _ = graph.run_task(crit_task_id, buffer, Some(critic_name.clone()))?;

            // Verify approval via trajectory matching
            if let Some(last) = buffer.last() {
                let critique_text = &last.thought;
                let display_len = 20.min(critique_text.len());
                info!(
                    "   Critic Feedback received ({}...)",
                    &critique_text[..display_len]
                );

                if critique_text.contains("APPROVED") {
                    info!("   Critic Approved! Stopping loop.");
                    let step = buffer.len() as u32 + 1;
                    buffer.add(TrajectoryPoint::new(
                        step,
                        "SelfCorrectionSuccess".to_string(),
                        "The solution was verified by the Critic.".to_string(),
                    ));
                    return Ok(());
                }

                if attempt == self.max_attempts {
                    warn!("   Max attempts reached. Returning last result.");
                    return Ok(());
                }

                // Feed critique back
                info!("   ♻️ Feeding critique back to Generator...");
                let step = buffer.len() as u32 + 1;
                buffer.add(TrajectoryPoint::new(
                    step,
                    "CritiqueFeedback".to_string(),
                    format!(
                        "Previous solution rejected. Critic said: {}. Please fix.",
                        critique_text
                    ),
                ));
            }
        }

        Ok(())
    }
}
