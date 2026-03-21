//! Swarm Pollination (RL-Tuned Information Brokerage)
//!
//! Handles peer-to-peer context sharing and behavioral RL tuning
//! driven by temporal difference expected value updates.

use pyo3::prelude::*;
use std::collections::HashMap;

/// RL state for an agent's information sharing behavior
#[derive(Clone, Debug)]
#[pyclass]
pub struct PollinatorState {
    /// Expected Value of sharing context (V(share)), mapped via TD loop
    pub raw_eagerness: f32,
    /// Bounded eagerness to share (0.0 to 1.0, Sigmoid mapped)
    #[pyo3(get)]
    pub share_probability: f32,
    
    // Tracking active pollinations for credit assignment
    // Map of (Agent_ID => Simulation Step it was shared)
    active_shares: HashMap<u32, u64>,
    
    /// Max ticks after a share event to attribute credit/blame
    #[pyo3(get, set)]
    pub recency_window: u64,

    /// Dilation weight scaling how heavily anomalous 'Surprise' dilates probability
    #[pyo3(get, set)]
    pub surprise_broadcast_weight: f32,

    /// Temperature governing how sensitive share_probability is to raw_eagerness
    #[pyo3(get, set)]
    pub sigmoid_temperature: f32,

    /// TD Learning rate (Alpha)
    #[pyo3(get, set)]
    pub alpha: f32,

    /// TD Future Discount factor (Gamma)
    #[pyo3(get, set)]
    pub gamma: f32,
}

#[pymethods]
impl PollinatorState {
    #[new]
    #[pyo3(signature = (recency_window=10, surprise_broadcast_weight=0.5, sigmoid_temperature=1.0, alpha=0.1, gamma=0.9))]
    pub fn new(recency_window: u64, surprise_broadcast_weight: f32, sigmoid_temperature: f32, alpha: f32, gamma: f32) -> Self {
        let mut state = PollinatorState {
            raw_eagerness: 0.0, // Neutral start
            share_probability: 0.5,
            active_shares: HashMap::new(),
            recency_window,
            surprise_broadcast_weight,
            sigmoid_temperature,
            alpha,
            gamma,
        };
        state.update_probability(0.0);
        state
    }

    /// Evaluates if this agent wants to share context based on its bounded probability.
    /// Incorporates the current state's `surprise_score` to dynamically dilate or suppress 
    /// the agent's baseline eagerness based on how locally volatile the environment is.
    pub fn should_pollinate(&self, random_val: f32, current_surprise: f32) -> bool {
        // High surprise (anomaly) temporarily boosts the willingness to broadcast context
        // This prevents the RL loop from suppressing emergency information flow
        // even if the agent normally leans "selfish" (low raw_eagerness)
        let effective_probability = (self.share_probability + (current_surprise * self.surprise_broadcast_weight)).min(1.0);
        random_val < effective_probability
    }

    /// Register that this agent shared context to a neighbor
    pub fn register_share(&mut self, target_agent_id: u32, current_step: u64) {
        self.active_shares.insert(target_agent_id, current_step);
    }

    /// Temporal Difference (TD) RL Reward Signal Hook
    /// Computes strictly isolated expected value updates (V) mitigating spam signals.
    pub fn apply_feedback(&mut self, target_agent_id: u32, reward_delta: f32, current_step: u64) {
        if let Some(share_time) = self.active_shares.get(&target_agent_id) {
            // Only claim credit if the share was recently enough
            if current_step.saturating_sub(*share_time) <= self.recency_window {
                
                let current_v = self.raw_eagerness;
                // Single-state loop approximation: next state expected value mirrors the updated baseline
                let next_v = current_v; 

                // V(s) <- V(s) + alpha * [R + gamma*V(s') - V(s)]
                let td_error = reward_delta + (self.gamma * next_v) - current_v;
                self.raw_eagerness += self.alpha * td_error;

                self.update_probability(0.0);
            }
            
            // Clear tracking after feedback resolved
            self.active_shares.remove(&target_agent_id);
        }
    }

    /// Retrieve copy of active sharing IDs to avoid concurrent locks when sending updates
    pub fn active_shares_keys(&self) -> Vec<u32> {
        self.active_shares.keys().cloned().collect()
    }

    /// Sigmoid function to bound the raw eagerness cleanly between 0 and 1
    fn update_probability(&mut self, _surprise_proxy: f32) {
        // Sigmoid mapping: 1 / (1 + e^(-x / T))
        self.share_probability = 1.0 / (1.0 + (-self.raw_eagerness / self.sigmoid_temperature).exp());
    }
}
