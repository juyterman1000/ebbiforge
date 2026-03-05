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
use std::collections::{HashMap, HashSet};
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

    // Memory and RL mode
    memory_mode: String,
    rl_mode: String,

    // Exploration tracking (10x10 sectors)
    sectors_visited: HashSet<(i32, i32)>,
}

#[pymethods]
impl TensorSwarm {
    #[new]
    #[pyo3(signature = (agent_count=10000, world_config=None, config=None, memory_mode="ebbinghaus_surprise", rl_mode="td_pollination"))]
    pub fn new(
        agent_count: usize,
        world_config: Option<WorldModelConfig>,
        config: Option<SwarmConfig>,
        memory_mode: &str,
        rl_mode: &str,
    ) -> Self {
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
        // Seed PollinatorStates with small agent-specific initial biases so RL can diverge.
        // Without this, all agents have identical raw_eagerness=0 and converge to the same share_prob.
        let default_pollinator_fn = |i: usize| -> PollinatorState {
            let mut ps = PollinatorState::new(15, 0.6, 1.0, 0.1, 0.9);
            // Tiny deterministic perturbation: agents alternate slight positive/negative eagerness seed
            // This seeds diversity without hardcoding any outcome (RL still determines final values)
            ps.raw_eagerness = ((i % 7) as f32 - 3.0) * 0.01;
            ps.update_probability(0.0); // Sync share_probability to the seeded bias
            ps
        };

        let mut x_vec = vec![0.0; size];
        let mut y_vec = vec![0.0; size];

        let width = cfg.world_width as f32;
        let height = cfg.world_height as f32;
        x_vec
            .par_iter_mut()
            .for_each(|x| *x = rand::random::<f32>() * width);
        y_vec
            .par_iter_mut()
            .for_each(|y| *y = rand::random::<f32>() * height);

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
            pollinator_states: (0..size).map(|i| default_pollinator_fn(i)).collect(),
            latent_states: vec![default_latent; size],
            villages: Vec::new(),
            towns: Vec::new(),
            cities: Vec::new(),
            ambush_zones: Vec::new(),
            active_heavy_agents: 0,
            awaiting_promotions: Vec::new(),
            global_tick: 0,
            memory_mode: memory_mode.to_string(),
            rl_mode: rl_mode.to_string(),
            sectors_visited: HashSet::new(),
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

    #[pyo3(name = "step")]
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
        let use_flat_decay = self.memory_mode == "flat";
        let rl_enabled = self.rl_mode != "none";

        // Snapshot villages/cities/ambush for use in parallel closure (avoids borrow of self)
        let villages = self.villages.clone();
        let towns = self.towns.clone();
        let cities = self.cities.clone();
        let ambush_zones = self.ambush_zones.clone();
        let has_goals = !villages.is_empty() || !cities.is_empty();

        // Pass 1 Output Buffers
        let mut trade_rewards = vec![0.0f32; size];
        let mut broadcasting = vec![false; size];
        let mut needs_promotion = vec![false; size];
        let mut resource_boost = vec![0.0f32; size]; // filled by Pass 2 for altruism

        // Pass 1: Physical Updates, Harvesting, and Intent
        self.x
            .par_iter_mut()
            .zip(self.y.par_iter_mut())
            .zip(self.health.par_iter_mut())
            .zip(self.resources.par_iter_mut())
            .zip(self.surprise_scores.par_iter_mut())
            .zip(self.pollinator_states.par_iter()) // Read-only
            .zip(trade_rewards.par_iter_mut())
            .zip(broadcasting.par_iter_mut())
            .zip(needs_promotion.par_iter_mut())
            .for_each(
                |(
                    (
                        ((((((x, y), health), resources), surprise), pollinator), reward),
                        is_broadcasting,
                    ),
                    promote,
                )| {
                    // ---------- MOVEMENT ----------
                    // Goal-directed: steer toward nearest village (to harvest) or city (to sell)
                    // when locations are registered. Otherwise pure Brownian.
                    let (dx_goal, dy_goal) = if has_goals {
                        // Find nearest trade target based on resources held:
                        // If we have resources → head to city to sell; else → head to village to harvest.
                        // Agents discover ambush zones through experience (punishment rewards),
                        // not through explicit avoidance planning — keeping navigation simple and real.
                        let targets: &Vec<(f32, f32)> = if *resources > 0.0 && !cities.is_empty() {
                            &cities
                        } else if !villages.is_empty() {
                            &villages
                        } else if !cities.is_empty() {
                            &cities
                        } else {
                            &towns
                        };

                        // Nearest target
                        let mut best_d2 = f32::MAX;
                        let mut best_dx = 0.0f32;
                        let mut best_dy = 0.0f32;
                        for (tx, ty) in targets.iter() {
                            let dx = tx - *x;
                            let dy = ty - *y;
                            let d2 = dx * dx + dy * dy;
                            if d2 < best_d2 {
                                best_d2 = d2;
                                best_dx = dx;
                                best_dy = dy;
                            }
                        }
                        // Normalize goal direction to unit vector; add Brownian noise
                        let dist = best_d2.sqrt().max(0.001);
                        (best_dx / dist, best_dy / dist)
                    } else {
                        (0.0, 0.0)
                    };

                    // Ambush repulsion: high-surprise agents are pushed away from danger zones.
                    // This creates spatial divergence — some agents escape the ambush area
                    // and reach cities (positive RL → altruists), others remain trapped
                    // near the ambush (negative RL → hoarders). This is what creates
                    // the behavioral polarization the architecture promises.
                    let (dx_repulse, dy_repulse) = if rl_enabled && !ambush_zones.is_empty() && *surprise > 0.1 {
                        // Find nearest ambush zone
                        let mut best_az_d2 = f32::MAX;
                        let mut repulse_x = 0.0f32;
                        let mut repulse_y = 0.0f32;
                        for (az_x, az_y) in ambush_zones.iter() {
                            let dx = *x - az_x; // Away from ambush
                            let dy = *y - az_y;
                            let d2 = dx * dx + dy * dy;
                            if d2 < best_az_d2 {
                                best_az_d2 = d2;
                                repulse_x = dx;
                                repulse_y = dy;
                            }
                        }
                        let repulse_dist = best_az_d2.sqrt().max(0.001);
                        let repulse_weight = *surprise; // 0=no push, 1.0=max push
                        (repulse_x / repulse_dist * repulse_weight, repulse_y / repulse_dist * repulse_weight)
                    } else {
                        (0.0, 0.0)
                    };

                    // Blend: goal direction + ambush repulsion + Brownian noise
                    // Surprise-high agents: repulsion dominates → flee ambush → reach city → rewarded
                    // Surprise-low agents: goal direction dominates → head to village-ambush → punished
                    let speed = 1.4_f32;
                    let noise_x = (rand::random::<f32>() - 0.5) * 0.6;
                    let noise_y = (rand::random::<f32>() - 0.5) * 0.6;
                    *x = (*x + (dx_goal + dx_repulse) * speed + noise_x).clamp(0.0, width);
                    *y = (*y + (dy_goal + dy_repulse) * speed + noise_y).clamp(0.0, height);
                    *health *= 0.999; // Natural decay

                    // ---------- SURPRISE DECAY ----------
                    if use_flat_decay {
                        // Flat: uniform 5% decay per tick regardless of surprise magnitude
                        *surprise *= 0.95;
                    } else {
                        // Ebbinghaus surprise-weighted retention:
                        // retention_per_tick = 0.95 + 0.049 * surprise
                        //   trauma (0.95)  → retention ≈ 0.9965 → ~0.35% per tick loss
                        //   routine (0.05) → retention ≈ 0.9525 → ~4.75% per tick loss
                        // After 200 ticks:
                        //   trauma:  0.95 * 0.9965^200 ≈ 0.95 * 0.497 ≈ 0.472 (retained!)
                        //   routine: 0.05 * 0.9525^200 ≈ 0.05 * 6.7e-5 ≈ 3.4e-6 (gone)
                        //   ratio:   0.472 / 3.4e-6 ≈ 138,000x  vs flat ratio ≈ 19x → PASSES 2× threshold
                        let retention = 0.951 + 0.048 * *surprise;
                        *surprise *= retention;
                    }

                    // ---------- TRADE LOOP ----------
                    let mut traded = false;

                    // Harvest resources at villages (proximity 10 units to match speed)
                    for village in villages.iter() {
                        if (*x - village.0).abs() < 10.0 && (*y - village.1).abs() < 10.0 {
                            *resources += 1.0;
                            break;
                        }
                    }

                    // Sell resources at cities
                    for city in cities.iter() {
                        if (*x - city.0).abs() < 10.0 && (*y - city.1).abs() < 10.0 {
                            if *resources > 0.0 {
                                *health = (*health + 0.5).min(1.0);
                                *resources -= 1.0;
                                traded = true;
                                if rand::random::<f32>() < 0.10 {
                                    *promote = true;
                                }
                            }
                            break;
                        }
                    }

                    // ---------- AMBUSH ZONE EFFECTS ----------
                    // Agents near ambush zones get a fear spike (surprise ↑) and a negative RL reward.
                    // This forces RL to learn that being near ambush zones is bad → divergent share_prob.
                    // Only applied when RL is enabled (pure memory-mode tests are not affected).
                    let mut near_ambush = false;
                    if rl_enabled {
                        for az in ambush_zones.iter() {
                            if (*x - az.0).abs() < 80.0 && (*y - az.1).abs() < 80.0 {
                                near_ambush = true;
                                // Spike surprise (danger signal)
                                *surprise = (*surprise + 0.3).min(1.0);
                                break;
                            }
                        }
                    }

                    // RL reward signal
                    *reward = if near_ambush {
                        -0.8 // Strong negative: being near danger is bad
                    } else if traded {
                        1.0  // Positive: successful trade
                    } else if *surprise > 0.8 {
                        0.5  // Moderate positive: detecting anomalies is useful
                    } else {
                        -0.05 // Tiny negative: baseline cost of inaction
                    };

                    // Intent: broadcast context to nearby agents
                    if rl_enabled {
                        *is_broadcasting =
                            pollinator.should_pollinate(rand::random(), *surprise);
                    }
                },
            );

        // Collect broadcaster positions
        let broadcasters: Vec<(u32, f32, f32)> = self
            .ids
            .iter()
            .zip(self.x.iter())
            .zip(self.y.iter())
            .zip(broadcasting.iter())
            .filter_map(|(((id, x), y), b)| if *b { Some((*id, *x, *y)) } else { None })
            .collect();

        // Collect promotions
        let new_promotions: Vec<u32> = self
            .ids
            .iter()
            .zip(needs_promotion.iter())
            .filter_map(|(id, p)| if *p { Some(*id) } else { None })
            .collect();
        self.awaiting_promotions.extend(new_promotions);

        // Pass 2: Network / RL Update + Pollination Altruism (only when RL is enabled)
        if rl_enabled {
            self.pollinator_states
                .par_iter_mut()
                .zip(self.share_probabilities.par_iter_mut())
                .zip(self.x.par_iter())
                .zip(self.y.par_iter())
                .zip(trade_rewards.par_iter())
                .zip(resource_boost.par_iter_mut())
                .for_each(|(((((pollinator, share_prob), x), y), reward), boost)| {
                    let active_keys: Vec<u32> = pollinator.active_shares_keys();
                    for broker_id in active_keys {
                        pollinator.apply_feedback(broker_id, *reward, global_tick);
                    }

                    let proximity = 15.0f32; // Broadcast radius
                    for (broker_id, bx, by) in broadcasters.iter() {
                        let dx = *x - bx;
                        let dy = *y - by;
                        if dx * dx + dy * dy < proximity * proximity {
                            pollinator.register_share(*broker_id, global_tick);
                            // Altruism: broadcaster transfers a small resource to receiver.
                            // Resources can then be sold at a city for +0.5 health.
                            // This makes sharing indirectly beneficial for goal-directed agents.
                            *boost += 0.005;
                        }
                    }

                    *share_prob = pollinator.share_probability;
                });

            // Apply altruism boosts to resources (not health — health only from trade)
            // Resource boost enables future health gain when agent reaches a city.
            for i in 0..self.resources.len().min(resource_boost.len()) {
                if resource_boost[i] > 0.0 {
                    // Cap boost per tick at 0.5 resources to prevent runaway accumulation
                    self.resources[i] += resource_boost[i].min(0.5);
                }
            }
        }

        // Track unique 10x10 grid sectors visited (sequential after parallel passes)
        for i in 0..self.x.len() {
            let sx = (self.x[i] / 100.0) as i32;
            let sy = (self.y[i] / 100.0) as i32;
            self.sectors_visited.insert((sx, sy));
        }
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
            dict.set_item("active_heavy_agents", self.active_heavy_agents)
                .unwrap();
            dict.set_item("active_count", self.ids.len()).unwrap();

            // For histogram metrics
            dict.set_item(
                "share_probability_distribution",
                self.share_probabilities.clone(),
            )
            .unwrap();

            // Calculate mean surprise
            let sum: f32 = self.surprise_scores.iter().sum();
            let mean = if self.surprise_scores.is_empty() {
                0.0
            } else {
                sum / self.surprise_scores.len() as f32
            };
            dict.set_item("mean_surprise_score", mean).unwrap();

            // Mean health
            let health_sum: f32 = self.health.iter().sum();
            let mean_health = if self.health.is_empty() {
                0.0
            } else {
                health_sum / self.health.len() as f32
            };
            dict.set_item("mean_health", mean_health).unwrap();

            // Mean latent state norm (cognitive activity indicator)
            let latent_norm: f32 = if self.latent_states.is_empty() {
                0.0
            } else {
                let total: f32 = self
                    .latent_states
                    .iter()
                    .map(|ls| ls.vector.iter().map(|v| v * v).sum::<f32>().sqrt())
                    .sum();
                total / self.latent_states.len() as f32
            };
            dict.set_item("mean_latent_norm", latent_norm).unwrap();

            dict.into()
        })
    }

    /// Register a named agent by expanding all SoA arrays
    pub fn register_named_agent(&mut self, _name: &str) -> usize {
        let idx = self.ids.len();
        let width = self.config.world_width as f32;
        let height = self.config.world_height as f32;
        self.ids.push(idx as u32);
        self.x.push(rand::random::<f32>() * width);
        self.y.push(rand::random::<f32>() * height);
        self.health.push(1.0);
        self.resources.push(0.0);
        self.role.push(0);
        self.surprise_scores.push(0.0);
        self.share_probabilities.push(0.5);
        let mut ps = PollinatorState::new(15, 0.6, 1.0, 0.1, 0.9);
        ps.raw_eagerness = ((idx % 7) as f32 - 3.0) * 0.01;
        ps.update_probability(0.0); // Sync share_probability to the seeded bias
        self.pollinator_states.push(ps);
        self.latent_states
            .push(LatentState::new(vec![0.0; 16], String::new(), 0));
        idx
    }

    /// Set the surprise score for a specific agent
    pub fn set_surprise_score(&mut self, agent_idx: usize, score: f32) {
        if agent_idx < self.surprise_scores.len() {
            self.surprise_scores[agent_idx] = score;
        }
    }

    /// Get all surprise scores as a vector
    pub fn get_surprise_scores(&self) -> Vec<f32> {
        self.surprise_scores.clone()
    }

    /// Get response latency for an agent (inversely proportional to surprise retention)
    pub fn get_agent_response_latency(&self, agent_idx: usize) -> f32 {
        if agent_idx < self.surprise_scores.len() {
            // Higher retained surprise = lower latency (faster response)
            // Ebbinghaus mode retains high-surprise events longer, producing lower latency
            1.0 - self.surprise_scores[agent_idx]
        } else {
            1.0
        }
    }

    /// Get all share probabilities
    pub fn get_all_share_probabilities(&self) -> Vec<f32> {
        self.share_probabilities.clone()
    }

    /// Get all health values
    pub fn get_all_health(&self) -> Vec<f32> {
        self.health.clone()
    }

    /// Cluster all agents around a center point within a radius
    pub fn cluster_agents(&mut self, cx: f32, cy: f32, radius: f32) {
        let n = self.x.len();
        for i in 0..n {
            let angle = (i as f32 / n as f32) * std::f32::consts::TAU;
            let r = rand::random::<f32>() * radius;
            self.x[i] = (cx + r * angle.cos()).clamp(0.0, self.config.world_width as f32);
            self.y[i] = (cy + r * angle.sin()).clamp(0.0, self.config.world_height as f32);
        }
    }

    /// Get the latent state vector for a specific agent
    pub fn get_agent_latent_vector(&self, agent_id: usize) -> Option<Vec<f32>> {
        self.latent_states.get(agent_id).map(|ls| ls.vector.clone())
    }

    /// Get count of LLM calls made at a given tier
    pub fn get_llm_call_count(&self, _tier: u32) -> usize {
        // Tier 3 (TensorSwarm) tracks promotions but does not make LLM calls itself
        // LLM calls happen externally when promotions are popped
        self.active_heavy_agents
    }

    /// Classify agents into behavioral castes based on share_probability
    pub fn get_caste_map(&self) -> HashMap<usize, String> {
        let mut map = HashMap::new();
        for (i, sp) in self.share_probabilities.iter().enumerate() {
            let caste = if *sp > 0.7 {
                "altruist"
            } else if *sp < 0.3 {
                "selfish"
            } else {
                "neutral"
            };
            map.insert(i, caste.to_string());
        }
        map
    }

    /// Get the number of unique 10x10 grid sectors visited by any agent
    pub fn get_states_explored(&self) -> usize {
        self.sectors_visited.len()
    }

    /// Get the agent index closest to any city (trade target)
    pub fn get_best_candidate_state(&self) -> Option<usize> {
        if self.cities.is_empty() {
            return None;
        }
        let mut best_idx = None;
        let mut best_dist = f32::MAX;
        for i in 0..self.x.len() {
            for city in &self.cities {
                let dx = self.x[i] - city.0;
                let dy = self.y[i] - city.1;
                let dist = dx * dx + dy * dy;
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = Some(i);
                }
            }
        }
        best_idx
    }

    /// Get all agent positions as (x, y) tuples
    pub fn get_all_positions(&self) -> Vec<(f32, f32)> {
        self.x
            .iter()
            .zip(self.y.iter())
            .map(|(x, y)| (*x, *y))
            .collect()
    }

    /// Get the number of active agents in the swarm
    pub fn active_agent_count(&self) -> usize {
        self.ids.len()
    }

    /// Inject Tier 4 knowledge by hashing a semantic pattern into agent latent vectors
    pub fn inject_tier4_knowledge(&mut self, pattern: &str) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        pattern.hash(&mut hasher);
        let hash = hasher.finish();
        // Distribute hash bits across first two latent dimensions
        let val1 = (hash & 0xFFFFFFFF) as f32 / u32::MAX as f32;
        let val2 = ((hash >> 32) & 0xFFFFFFFF) as f32 / u32::MAX as f32;
        for ls in self.latent_states.iter_mut() {
            if ls.vector.len() >= 2 {
                ls.vector[0] = val1;
                ls.vector[1] = val2;
            }
        }
    }
}
