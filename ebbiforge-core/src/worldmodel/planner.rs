use super::{ActionScore, AutoregressivePredictor, LatentEncoder, LatentState, WorldModelConfig};
use pyo3::prelude::*;
use tracing::info;

/// Planning engine for action selection via mental simulation
#[pyclass]
pub struct PlanningEngine {
    config: WorldModelConfig,
    encoder: LatentEncoder,
    predictor: AutoregressivePredictor,
}

#[pymethods]
impl PlanningEngine {
    #[new]
    #[pyo3(signature = (config = None))]
    pub fn new(config: Option<WorldModelConfig>) -> Self {
        let cfg = config.clone().unwrap_or_default();
        PlanningEngine {
            config: cfg.clone(),
            encoder: LatentEncoder::new(config.clone()).unwrap(),
            predictor: AutoregressivePredictor::new(config).unwrap(),
        }
    }

    /// Plan the best action given current state and goal
    pub fn plan(
        &self,
        current_state: &LatentState,
        candidate_actions: Vec<String>,
        goal: String,
    ) -> ActionScore {
        // Encode semantic goal target
        let goal_encoding = self.encoder.encode_action(goal.clone());
        let goal_state = LatentState::new(goal_encoding, "goal".to_string(), 0);

        let mut best_action = String::new();
        let mut best_score = f32::NEG_INFINITY;
        let mut best_outcome = String::new();

        info!("[Planner] Planning for goal: '{}'", goal);

        for action in &candidate_actions {
            // Encode natural language action
            let action_encoding = self.encoder.encode_action(action.clone());

            // Rollout future (State-of-the-art prediction)
            let prediction = self.predictor.rollout(
                current_state,
                action_encoding,
                self.config.prediction_steps,
            );

            // Score: Semantic similarity of final state to goal
            // Both state and goal are language-conditioned
            let final_state = prediction.future_states.last();
            let score = match final_state {
                Some(state) => state.similarity(&goal_state) * prediction.confidence,
                None => 0.0,
            };

            if score > best_score {
                best_score = score;
                best_action = action.clone();
                best_outcome = format!(
                    "Predicted {} steps, goal align: {:.3}",
                    prediction.future_states.len(),
                    score
                );
            }
        }

        info!("Best action: {} (score: {:.3})", best_action, best_score);

        ActionScore {
            action: best_action,
            score: best_score,
            predicted_outcome: best_outcome,
        }
    }

    /// Evaluate all actions and return ranked scores
    pub fn evaluate_actions(
        &self,
        current_state: &LatentState,
        candidate_actions: Vec<String>,
        goal: String,
    ) -> Vec<ActionScore> {
        let goal_encoding = self.encoder.encode_action(goal);
        let goal_state = LatentState::new(goal_encoding, "goal".to_string(), 0);

        let mut scores: Vec<ActionScore> = candidate_actions
            .iter()
            .map(|action| {
                let action_encoding = self.encoder.encode_action(action.clone());
                let prediction = self.predictor.rollout(
                    current_state,
                    action_encoding,
                    self.config.prediction_steps,
                );

                let score = prediction
                    .future_states
                    .last()
                    .map(|s| s.similarity(&goal_state) * prediction.confidence)
                    .unwrap_or(0.0);

                ActionScore {
                    action: action.clone(),
                    score,
                    predicted_outcome: format!("P(align)={:.3}", score),
                }
            })
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        scores
    }
}
