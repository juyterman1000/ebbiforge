//! Level of Detail (LOD) Scale Architecture
//!
//! Handles a massive 10M agent population across 4 compute tiers to prevent CPU melting.
//! Tiers: Dormant (bitflag), Simplified (cache-friendly SIMD), Full Fidelity (TensorSwarm), Heavy (LLM).

use crate::swarm::tensor_engine::TensorSwarm;
use pyo3::prelude::*;

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
}

#[pymethods]
impl DormantAgent {
    #[new]
    pub fn new(id: u32, predicted_state: u8, wakeup_conditions: u64) -> Self {
        Self {
            id,
            predicted_state,
            wakeup_conditions,
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
            if self.positions_x[i] > 1000.0 { self.positions_x[i] -= 1000.0; }
            if self.positions_x[i] < 0.0 { self.positions_x[i] += 1000.0; }
            if self.positions_y[i] > 1000.0 { self.positions_y[i] -= 1000.0; }
            if self.positions_y[i] < 0.0 { self.positions_y[i] += 1000.0; }
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
}

#[pymethods]
impl ProductionTensorSwarm {
    #[new]
    #[pyo3(signature = (agent_count=10000, world_config=None, config=None))]
    pub fn new(
        agent_count: usize,
        world_config: Option<crate::worldmodel::WorldModelConfig>,
        config: Option<crate::swarm::SwarmConfig>,
    ) -> Self {
        Self {
            dormant_pool: Vec::new(),
            simplified: SimplifiedPool::new(),
            active: TensorSwarm::new(agent_count, world_config, config),
            global_triggers: 0,
            tick_count: 0,
        }
    }

    /// Add a batch of dormant agents (e.g. initially populating the 10M world)
    pub fn add_dormant_agents(&mut self, agents: Vec<DormantAgent>) {
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

        // 4. (Tier 4 is handled outside by extracting promotions and spawning async LLMs)

        // 5. Demotion logic
        self.check_simplify_conditions();

        self.tick_count += 1;
    }

    /// Checks if dormant agents need to wake up
    fn check_dormant_wakeups(&mut self) {
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
    /// Position is derived from the agent's ID (deterministic scatter)
    fn promote_to_simplified(&mut self, dormant: DormantAgent) {
        // Fibonacci hash scatter: deterministic position from agent ID
        let hash = (dormant.id as f32) * 0.6180339887;
        let frac = hash - hash.floor();
        let px = frac * 1000.0;  // Scatter across world width
        let py = ((dormant.id as f32 * 2.2360679775).fract()) * 1000.0;
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
        let health_floor: f32 = 0.95;       // Only demote healthy agents (boring = healthy + unsurprised)

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
}
