//! Level of Detail (LOD) Scale Architecture
//!
//! Handles a massive 10M agent population across 4 compute tiers to prevent CPU melting.
//! Tiers: Dormant (bitflag), Simplified (cache-friendly SIMD), Full Fidelity (TensorSwarm), Heavy (LLM).

use crate::swarm::tensor_engine::TensorSwarm;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Tier 1: Dormant Agent
/// Represents an agent that is far from interesting events.
/// Requires minimal processing (0.01ms per tick cost)
#[derive(Clone, Debug)]
#[pyclass]
pub struct DormantAgent {
    #[pyo3(get, set)]
    pub id: u32,
    #[pyo3(get, set)]
    pub predicted_state: u8,
    #[pyo3(get, set)]
    pub wakeup_conditions: u64, // Bitflags for triggers
    #[pyo3(get, set)]
    pub x: f32,
    #[pyo3(get, set)]
    pub y: f32,
}

#[pymethods]
impl DormantAgent {
    #[new]
    #[pyo3(signature = (id, predicted_state, wakeup_conditions, x=0.0, y=0.0))]
    pub fn new(id: u32, predicted_state: u8, wakeup_conditions: u64, x: f32, y: f32) -> Self {
        Self {
            id,
            predicted_state,
            wakeup_conditions,
            x,
            y,
        }
    }
}

/// Tier 2: Simplified Physics Pool
/// Fast cache-coherent layout for minimal spatial updates (10Hz).
/// Updates only positions, velocities, and basic state.
#[pyclass]
pub struct SimplifiedPool {
    #[pyo3(get, set)]
    pub positions_x: Vec<f32>,
    #[pyo3(get, set)]
    pub positions_y: Vec<f32>,
    #[pyo3(get, set)]
    pub velocities_x: Vec<f32>,
    #[pyo3(get, set)]
    pub velocities_y: Vec<f32>,
    #[pyo3(get, set)]
    pub states: Vec<u8>,
}

impl SimplifiedPool {
    pub fn new() -> Self {
        Self {
            positions_x: Vec::new(),
            positions_y: Vec::new(),
            velocities_x: Vec::new(),
            velocities_y: Vec::new(),
            states: Vec::new(),
        }
    }

    pub fn update_batch(&mut self) {
        let len = self.positions_x.len();
        for i in 0..len {
            self.positions_x[i] += self.velocities_x[i];
            self.positions_y[i] += self.velocities_y[i];
            // Wrap at world boundaries (1000x1000 default)
            if self.positions_x[i] > 1000.0 {
                self.positions_x[i] -= 1000.0;
            }
            if self.positions_x[i] < 0.0 {
                self.positions_x[i] += 1000.0;
            }
            if self.positions_y[i] > 1000.0 {
                self.positions_y[i] -= 1000.0;
            }
            if self.positions_y[i] < 0.0 {
                self.positions_y[i] += 1000.0;
            }
        }
    }
}

/// The Orchestrator of the 4-Tier Scale Architecture
#[pyclass]
pub struct ProductionTensorSwarm {
    // Tier 1
    dormant_pool: Vec<DormantAgent>,

    // Tier 2
    simplified: SimplifiedPool,

    // Tier 3
    pub active: TensorSwarm,

    // Tier 4 (Handled externally via active.awaiting_promotions -> AgentGraph async)

    // Global conditions (e.g. ambient surprise/danger) for awakening dormant agents
    global_triggers: u64,

    // Global simulation clock
    pub tick_count: u64,

    // Wavefront propagation state
    wavefront_center: Option<(f32, f32)>,
    wavefront_radius: f32,

    // Track initial dormant count for coverage calculation
    initial_dormant_count: usize,
}

#[pymethods]
impl ProductionTensorSwarm {
    #[new]
    #[pyo3(signature = (agent_count=10000, world_config=None, config=None, memory_mode="ebbinghaus_surprise", rl_mode="td_pollination"))]
    pub fn new(
        agent_count: usize,
        world_config: Option<crate::worldmodel::WorldModelConfig>,
        config: Option<crate::swarm::SwarmConfig>,
        memory_mode: &str,
        rl_mode: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            dormant_pool: Vec::new(),
            simplified: SimplifiedPool::new(),
            active: TensorSwarm::new(agent_count, world_config, config, memory_mode, rl_mode)?,
            global_triggers: 0,
            tick_count: 0,
            wavefront_center: None,
            wavefront_radius: 0.0,
            initial_dormant_count: 0,
        })
    }

    /// Add a batch of dormant agents (e.g. initially populating the 10M world)
    pub fn add_dormant_agents(&mut self, agents: Vec<DormantAgent>) {
        self.initial_dormant_count += agents.len();
        self.dormant_pool.extend(agents);
    }

    /// Set global environmental triggers (using bitflags)
    pub fn set_global_triggers(&mut self, triggers: u64) {
        self.global_triggers = triggers;
    }

    /// Primary execution loop. Distributes clock cycles across the Tiers.
    pub fn tick(&mut self) {
        // 1. Update dormant agents (extremely fast bitflag checks)
        self.check_dormant_wakeups();

        // 2. Simplified physics (10 Hz)
        if self.tick_count % 10 == 0 {
            self.simplified.update_batch();
        }

        // 3. Full simulation (100 Hz / Every Tick)
        self.active.step();

        // 3.5 Expand wavefront if active — wake dormant agents progressively
        if let Some(center) = self.wavefront_center {
            self.wavefront_radius += 15.0; // 15 units per tick expansion rate
            let r2 = self.wavefront_radius * self.wavefront_radius;

            // Apply environmental shock at wavefront edge (internal: params always valid)
            let _ = self
                .active
                .apply_environmental_shock(center, self.wavefront_radius, 0.8);

            // Wake dormant agents within wavefront radius
            let mut i = 0;
            while i < self.dormant_pool.len() {
                let dx = self.dormant_pool[i].x - center.0;
                let dy = self.dormant_pool[i].y - center.1;
                if dx * dx + dy * dy <= r2 {
                    let agent = self.dormant_pool[i].clone();
                    self.promote_to_simplified(agent);
                    self.dormant_pool.swap_remove(i);
                } else {
                    i += 1;
                }
            }
        }

        // 4. Demotion logic
        self.check_simplify_conditions();

        self.tick_count += 1;
    }

    /// Checks if dormant agents need to wake up via global triggers.
    /// When a wavefront is active, global-trigger wakeups are suppressed so that
    /// agents only wake progressively as the spatial wavefront reaches them.
    fn check_dormant_wakeups(&mut self) {
        // If a signal wavefront is propagating, skip global bitflag wakeup —
        // agents are woken spatially by the wavefront expansion instead.
        if self.wavefront_center.is_some() {
            return;
        }

        let mut i = 0;
        while i < self.dormant_pool.len() {
            let agent = &self.dormant_pool[i];
            // Bitwise check: if the agent's wakeup conditions overlap with global triggers
            if (agent.wakeup_conditions & self.global_triggers) != 0 {
                let agent_copy = agent.clone();
                self.promote_to_simplified(agent_copy);
                self.dormant_pool.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    /// Promote a Dormant agent into the Simplified Pool
    /// Uses the agent's stored position if available, otherwise deterministic scatter
    fn promote_to_simplified(&mut self, dormant: DormantAgent) {
        let px = if dormant.x != 0.0 || dormant.y != 0.0 {
            dormant.x
        } else {
            // Fibonacci hash scatter for agents without explicit position
            let hash = (dormant.id as f32) * 0.6180339887;
            let frac = hash - hash.floor();
            frac * 1000.0
        };
        let py = if dormant.x != 0.0 || dormant.y != 0.0 {
            dormant.y
        } else {
            ((dormant.id as f32 * 2.2360679775).fract()) * 1000.0
        };
        // Velocity from predicted_state: higher state = faster
        let speed = 0.05 + (dormant.predicted_state as f32) * 0.02;
        self.simplified.positions_x.push(px);
        self.simplified.positions_y.push(py);
        self.simplified.velocities_x.push(speed);
        self.simplified.velocities_y.push(speed * 0.7);
        self.simplified.states.push(dormant.predicted_state);
    }

    /// Demote boring Tier 3 agents to Tier 2 (Simplified).
    /// An agent is "boring" if its surprise score is below the demotion threshold
    /// for a sustained period. This frees Tier 3 compute for interesting agents.
    fn check_simplify_conditions(&mut self) {
        let surprise_scores = &self.active.surprise_scores;
        let health_scores = &self.active.health;
        let demotion_threshold: f32 = 0.01; // Demote if surprise < 1%
        let health_floor: f32 = 0.95; // Only demote healthy agents (boring = healthy + unsurprised)

        // Scan Tier 3 for agents that are both healthy and unsurprised
        // These agents are wasting Tier 3 compute cycles
        let mut demote_count: usize = 0;
        let n = surprise_scores.len().min(health_scores.len());
        for i in 0..n {
            if surprise_scores[i] < demotion_threshold && health_scores[i] > health_floor {
                demote_count += 1;
            }
        }

        // Move boring agents into Simplified pool (batch)
        // We demote at most 1% of the population per tick to avoid thrashing
        let max_demotions = n / 100;
        let actual_demotions = demote_count.min(max_demotions);
        for i in 0..actual_demotions {
            if i < n {
                self.simplified.positions_x.push(self.active.x[i]);
                self.simplified.positions_y.push(self.active.y[i]);
                self.simplified.velocities_x.push(0.05);
                self.simplified.velocities_y.push(0.05);
                self.simplified.states.push(0);
            }
        }
    }

    /// Bridge to let Python extract waiting LLM operations exactly as before
    pub fn pop_promotions(&mut self) -> Vec<u32> {
        self.active.pop_promotions()
    }

    // ════════════════════════════════════════════════════════════════════════
    // Forwarding methods to TensorSwarm (Tier 3)
    // ════════════════════════════════════════════════════════════════════════

    /// Register a named agent in the active swarm, returning its index
    pub fn register_named_agent(&mut self, name: &str) -> usize {
        self.active.register_named_agent(name)
    }

    /// Set the surprise score for a specific agent
    pub fn set_surprise_score(&mut self, agent_idx: usize, score: f32) -> PyResult<()> {
        self.active.set_surprise_score(agent_idx, score)
    }

    /// Get all surprise scores
    pub fn get_surprise_scores(&self) -> Vec<f32> {
        self.active.get_surprise_scores()
    }

    /// Get response latency for an agent
    pub fn get_agent_response_latency(&self, agent_idx: usize) -> PyResult<f32> {
        self.active.get_agent_response_latency(agent_idx)
    }

    /// Get all share probabilities
    pub fn get_all_share_probabilities(&self) -> Vec<f32> {
        self.active.get_all_share_probabilities()
    }

    /// Get all health values
    pub fn get_all_health(&self) -> Vec<f32> {
        self.active.get_all_health()
    }

    /// Cluster all agents around a center point
    pub fn cluster_agents(&mut self, cx: f32, cy: f32, radius: f32) {
        self.active.cluster_agents(cx, cy, radius);
    }

    /// Register spatial locations for the simulation
    pub fn register_locations(
        &mut self,
        villages: Vec<(f32, f32)>,
        towns: Vec<(f32, f32)>,
        cities: Vec<(f32, f32)>,
        ambush_zones: Vec<(f32, f32)>,
    ) {
        self.active
            .register_locations(villages, towns, cities, ambush_zones);
    }

    /// Inject Tier 4 knowledge into agent latent vectors
    pub fn inject_tier4_knowledge(&mut self, pattern: &str) {
        self.active.inject_tier4_knowledge(pattern);
    }

    /// Trigger a signal wavefront expanding from a center point
    pub fn trigger_signal_wavefront(&mut self, center: (f32, f32)) {
        self.wavefront_center = Some(center);
        self.wavefront_radius = 0.0;
        // Apply initial shock at the epicenter
        let _ = self.active.apply_environmental_shock(center, 15.0, 1.0);
    }

    /// Get the fraction of dormant agents that have been awakened by the wavefront
    pub fn get_propagation_coverage(&self) -> f32 {
        if self.initial_dormant_count == 0 {
            return 0.0;
        }
        let awakened = self.initial_dormant_count - self.dormant_pool.len();
        awakened as f32 / self.initial_dormant_count as f32
    }

    /// Get the latent state vector for a specific agent
    pub fn get_agent_latent_vector(&self, agent_id: usize) -> Option<Vec<f32>> {
        self.active.get_agent_latent_vector(agent_id)
    }

    /// Get the count of LLM calls at a given tier
    pub fn get_llm_call_count(&self, tier: u32) -> usize {
        self.active.get_llm_call_count(tier)
    }

    /// Get agent caste classification map
    pub fn get_caste_map(&self) -> HashMap<usize, String> {
        self.active.get_caste_map()
    }

    /// Get number of unique spatial sectors explored
    pub fn get_states_explored(&self) -> usize {
        self.active.get_states_explored()
    }

    /// Get the agent closest to any trade target
    pub fn get_best_candidate_state(&self) -> Option<usize> {
        self.active.get_best_candidate_state()
    }

    /// Get all agent positions
    pub fn get_all_positions(&self) -> Vec<(f32, f32)> {
        self.active.get_all_positions()
    }

    /// Get count of active agents
    pub fn active_agent_count(&self) -> usize {
        self.active.active_agent_count()
    }

    /// Get population metrics snapshot
    pub fn sample_population_metrics(&self) -> PyObject {
        self.active.sample_population_metrics()
    }
}
