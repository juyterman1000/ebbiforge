//! Tensor-Based Swarm Engine
//!
//! Uses Struct-of-Arrays (SoA) layout for cache-friendly updates of millions of agents.
//! Simulates GPU-like batch processing on CPU using Rayon.

use super::SwarmConfig;
use crate::swarm::pollination::PollinatorState;
use crate::worldmodel::{LatentState, WorldModelConfig};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use tracing::info;

/// Massive Swarm using SoA (Tensor) layout
#[pyclass]
pub struct TensorSwarm {
    config: SwarmConfig,
    // Tensor Columns (Vectors)
    pub ids: Vec<u32>,
    #[pyo3(get)]
    pub x: Vec<f32>,
    #[pyo3(get)]
    pub y: Vec<f32>,
    #[pyo3(get)]
    pub health: Vec<f32>,
    #[pyo3(get)]
    pub resources: Vec<f32>,
    #[pyo3(get)]
    pub role: Vec<u8>,

    // Cognitive / RL State
    #[pyo3(get)]
    pub surprise_scores: Vec<f32>,
    #[pyo3(get)]
    pub share_probabilities: Vec<f32>,
    
    // Rust-internal Tracking
    pollinator_states: Vec<PollinatorState>,
    latent_states: Vec<LatentState>,

    // Spatial Memory
    villages: Vec<(f32, f32)>,
    towns: Vec<(f32, f32)>,
    cities: Vec<(f32, f32)>,
    ambush_zones: Vec<(f32, f32)>,

    // Analytics
    active_heavy_agents: usize,
    pub awaiting_promotions: Vec<u32>,

    // Time Tracking
    pub global_tick: u64,
}

#[pymethods]
impl TensorSwarm {
    #[new]
    #[pyo3(signature = (agent_count=10000, world_config=None, config=None))]
    pub fn new(agent_count: usize, world_config: Option<WorldModelConfig>, config: Option<SwarmConfig>) -> Self {
        let mut cfg = config.unwrap_or_default();
        cfg.population_size = agent_count;
        let w_cfg = world_config.unwrap_or_default();
        
        // Sync SwarmConfig bounds to WorldModelConfig grid
        cfg.world_width = w_cfg.grid_size.0;
        cfg.world_height = w_cfg.grid_size.1;

        let size = cfg.population_size;

        info!(
            "🌐 [Swarm] Initializing tensor store for {} agents...",
            size
        );

        let default_latent = LatentState::new(vec![0.0; w_cfg.latent_dim], "".to_string(), 0);
        let default_pollinator = PollinatorState::new(15, 0.6, 1.0, 0.1, 0.9);

        let mut x_vec = vec![0.0; size];
        let mut y_vec = vec![0.0; size];
        
        let width = cfg.world_width as f32;
        let height = cfg.world_height as f32;
        x_vec.par_iter_mut().for_each(|x| *x = rand::random::<f32>() * width);
        y_vec.par_iter_mut().for_each(|y| *y = rand::random::<f32>() * height);

        // Initialize vectors (SoA)
        TensorSwarm {
            config: cfg,
            ids: (0..size as u32).collect(),
            x: x_vec,
            y: y_vec,
            health: vec![1.0; size],
            resources: vec![0.0; size],
            role: vec![0; size], // 0=Worker, 1=Scout, etc.
            surprise_scores: vec![0.0; size],
            share_probabilities: vec![0.5; size],
            pollinator_states: vec![default_pollinator; size],
            latent_states: vec![default_latent; size],
            villages: Vec::new(),
            towns: Vec::new(),
            cities: Vec::new(),
            ambush_zones: Vec::new(),
            active_heavy_agents: 0,
            awaiting_promotions: Vec::new(),
            global_tick: 0,
        }
    }

    /// Initialize positions randomly
    pub fn randomize_positions(&mut self) {
        let width = self.config.world_width as f32;
        let height = self.config.world_height as f32;

        // Parallel init
        self.x
            .par_iter_mut()
            .for_each(|x| *x = rand::random::<f32>() * width);
        self.y
            .par_iter_mut()
            .for_each(|y| *y = rand::random::<f32>() * height);
    }

    #[pyo3(name="step")]
    pub fn step(&mut self) {
        self.tick();
    }

    /// Execute a simulation step (Batch Update with RL and Memory physics)
    pub fn tick(&mut self) {
        self.global_tick += 1;
        let global_tick = self.global_tick;
        
        let width = self.config.world_width as f32;
        let height = self.config.world_height as f32;
        let size = self.ids.len();

        // Pass 1 Output Buffers
        let mut trade_rewards = vec![0.0; size];
        let mut broadcasting = vec![false; size];
        let mut needs_promotion = vec![false; size];

        // Pass 1: Physical Updates, Harvesting, and Intent
        self.x
            .par_iter_mut()
            .zip(self.y.par_iter_mut())
            .zip(self.health.par_iter_mut())
            .zip(self.resources.par_iter_mut())
            .zip(self.surprise_scores.par_iter_mut())
            .zip(self.pollinator_states.par_iter()) // Read-only access to intent
            .zip(trade_rewards.par_iter_mut())
            .zip(broadcasting.par_iter_mut())
            .zip(needs_promotion.par_iter_mut())
            .for_each(|((((((((x, y), health), resources), surprise), pollinator), reward), is_broadcasting), promote)| {
                // Rule: Brownian Motion
                *x = (*x + (rand::random::<f32>() - 0.5) * 2.0).clamp(0.0, width);
                *y = (*y + (rand::random::<f32>() - 0.5) * 2.0).clamp(0.0, height);
                *health *= 0.999; // Natural decay

                // Rule: Ebbinghaus decay on surprise_score
                let retention = (-0.1 * (1.0 - *surprise).max(0.1)).exp();
                *surprise = *surprise * retention;

                let mut traded = false;

                // Harvest resources at villages
                for village in self.villages.iter() {
                    if (*x - village.0).abs() < 5.0 && (*y - village.1).abs() < 5.0 {
                        *resources += 1.0;
                        break;
                    }
                }

                // Sell resources at cities
                for city in self.cities.iter() {
                    if (*x - city.0).abs() < 5.0 && (*y - city.1).abs() < 5.0 {
                        if *resources > 0.0 {
                            *health = (*health + 0.5).min(1.0); // Heal from successful trade
                            *resources -= 1.0;
                            traded = true;
                            // Signal that a complex trade occurred, triggering LLM negotiation 10% of the time
                            if rand::random::<f32>() < 0.10 {
                                *promote = true;
                            }
                        }
                        break;
                    }
                }

                // RL Signal: A successful trade validates any past info we acted on.
                // We waste a tiny bit of energy if we didn't trade (baseline survival cost).
                *reward = if traded || *surprise > 0.8 { 1.0 } else { -0.1 };

                // Determine if we INTEND to share our context to local neighbors
                *is_broadcasting = pollinator.should_pollinate(rand::random(), *surprise);
            });

        // Optimization: Collect the spatial coordinates of ONLY the agents who decided to broadcast
        // This avoids N^2 distance checks. 
        let broadcasters: Vec<(u32, f32, f32)> = self.ids.iter().zip(self.x.iter()).zip(self.y.iter()).zip(broadcasting.iter())
            .filter_map(|(((id, x), y), b)| if *b { Some((*id, *x, *y)) } else { None })
            .collect();
            
        // Collect promotions
        let new_promotions: Vec<u32> = self.ids.iter().zip(needs_promotion.iter())
            .filter_map(|(id, p)| if *p { Some(*id) } else { None })
            .collect();
        self.awaiting_promotions.extend(new_promotions);

        // Pass 2: Network / RL Update
        // We apply the physical reward to the RL engine (TD(0) update mapping back to info-brokers),
        // and register new info-brokers if we are near any broadcasters.
        self.pollinator_states
            .par_iter_mut()
            .zip(self.share_probabilities.par_iter_mut())
            .zip(self.x.par_iter())
            .zip(self.y.par_iter())
            .zip(trade_rewards.par_iter())
            .for_each(|((((pollinator, share_prob), x), y), reward)| {
                
                // 1. Send the trade reward feedback back to whoever shared context with us recently
                // The pollinator state holds a hashmap of (Agent_ID -> Tick_of_Share)
                // We use `.clone()` on the keys to avoid concurrent borrow mutations while sending feedback
                let active_keys: Vec<u32> = pollinator.active_shares_keys(); 
                for broker_id in active_keys {
                     pollinator.apply_feedback(broker_id, *reward, global_tick);
                }

                // 2. Receive new signals from nearby broadcasters (Simulating P2P Info Exchange)
                // If an agent is broadcasting within D=5.0, we "hear" them and credit them later if we trade
                for (broker_id, bx, by) in broadcasters.iter() {
                    if (*x - bx).abs() < 5.0 && (*y - by).abs() < 5.0 {
                        pollinator.register_share(*broker_id, global_tick);
                    }
                }

                *share_prob = pollinator.share_probability;
            });
    }

    /// Get state of a specific agent (for Promotion)
    pub fn get_agent_state(&self, id: usize) -> Option<(f32, f32, f32)> {
        if id < self.ids.len() {
            Some((self.x[id], self.y[id], self.health[id]))
        } else {
            None
        }
    }

    /// Retrieve the queue of agents that reached a promotion trigger, clearing it
    pub fn pop_promotions(&mut self) -> Vec<u32> {
        let promoted = self.awaiting_promotions.clone();
        self.active_heavy_agents += promoted.len(); // Track total spawned
        self.awaiting_promotions.clear();
        promoted
    }

    /// Register tracking locations for the spatial simulation
    pub fn register_locations(
        &mut self,
        villages: Vec<(f32, f32)>,
        towns: Vec<(f32, f32)>,
        cities: Vec<(f32, f32)>,
        ambush_zones: Vec<(f32, f32)>,
    ) {
        self.villages = villages;
        self.towns = towns;
        self.cities = cities;
        self.ambush_zones = ambush_zones;
    }

    /// Force high surprise score on agents within a blast radius
    pub fn apply_environmental_shock(&mut self, location: (f32, f32), radius: f32, intensity: f32) {
        let r2 = radius * radius;
        self.x
            .par_iter_mut()
            .zip(self.y.par_iter_mut())
            .zip(self.surprise_scores.par_iter_mut())
            .for_each(|((x, y), surprise)| {
                let dx = *x - location.0;
                let dy = *y - location.1;
                if (dx * dx + dy * dy) <= r2 {
                    // Pull everybody to that exact zone immediately and set their surprise
                    *surprise = intensity; 
                }
            });
    }

    /// Provide standard simulation metrics snapshot
    pub fn sample_population_metrics(&self) -> PyObject {
        Python::with_gil(|py| {
            let dict = PyDict::new_bound(py);
            dict.set_item("active_heavy_agents", self.active_heavy_agents).unwrap();
            
            // For histogram metrics
            dict.set_item("share_probability_distribution", self.share_probabilities.clone()).unwrap();
            
            // Calculate mean surprise
            let sum: f32 = self.surprise_scores.iter().sum();
            let mean = if self.surprise_scores.is_empty() { 0.0 } else { sum / self.surprise_scores.len() as f32 };
            dict.set_item("mean_surprise_score", mean).unwrap();

            // Mean latent state norm (cognitive activity indicator)
            let latent_norm: f32 = if self.latent_states.is_empty() {
                0.0
            } else {
                let total: f32 = self.latent_states.iter()
                    .map(|ls| ls.vector.iter().map(|v| v * v).sum::<f32>().sqrt())
                    .sum();
                total / self.latent_states.len() as f32
            };
            dict.set_item("mean_latent_norm", latent_norm).unwrap();
            
            dict.into()
        })
    }
}
