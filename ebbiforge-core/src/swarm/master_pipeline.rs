use super::grid::SpatialHashGrid;
use super::mmap_pool::MmapSwarmPool;
use super::pheromone::PheromoneField;
use rand::Rng;
use std::time::Instant;

/// Runtime-tunable SIRS propagation parameters.
///
/// R₀_base = transfer / (1 - decay). Default: 0.08 / 0.08 = 1.0 (critical).
/// Refractory negative feedback drives R₀_eff below R₀_base when many agents
/// are infected, creating self-organized criticality.
#[derive(Clone, Debug)]
pub struct PropagationConfig {
    /// Per-tick multiplicative decay of surprise (0..1). Higher = slower decay.
    pub surprise_decay: f32,
    /// Fraction of weighted neighbor surprise absorbed per tick.
    pub surprise_transfer: f32,
    /// Distance weighting exponent: w = (1 - d/R)^falloff. 1.0 = linear, 2.0 = quadratic.
    pub distance_falloff: f32,
    /// Surprise level above which refractory immunity builds.
    pub refractory_threshold: f32,
    /// Rate at which refractory state accumulates while surprised.
    pub refractory_buildup: f32,
    /// Per-tick multiplicative decay of refractory state when not surprised.
    pub refractory_decay: f32,
    /// How strongly danger pheromone (CH_1) reignites surprise.
    pub danger_feedback: f32,
    /// Surprise threshold for emitting novelty beacon (CH_4).
    pub novelty_emission: f32,
    /// Steering weight toward novelty pheromone gradient.
    pub novelty_attraction: f32,
    /// Surprise threshold for emitting danger pheromone (CH_1).
    pub danger_emission_threshold: f32,
}

impl Default for PropagationConfig {
    fn default() -> Self {
        Self {
            surprise_decay: 0.92,
            surprise_transfer: 0.08,
            distance_falloff: 1.0,
            refractory_threshold: 0.6,
            refractory_buildup: 0.3,
            refractory_decay: 0.98,
            danger_feedback: 0.15,
            novelty_emission: 0.5,
            novelty_attraction: 0.2,
            danger_emission_threshold: 0.3,
        }
    }
}

impl PropagationConfig {
    /// Basic reproduction number (no refractory). R₀ > 1 = supercritical.
    pub fn r0_base(&self) -> f32 {
        self.surprise_transfer / (1.0 - self.surprise_decay)
    }

    /// Effective R₀ given mean refractory level across the population.
    pub fn r0_effective(&self, mean_refractory: f32) -> f32 {
        self.r0_base() * (1.0 - mean_refractory)
    }
}

/// Gene ranges for clamping after mutation. (min, max) pairs.
pub const GENE_RANGES: [(f32, f32); 6] = [
    (0.80, 0.99), // gene_decay
    (0.01, 0.30), // gene_transfer
    (0.05, 0.80), // gene_refractory
    (0.00, 0.50), // gene_danger_sense
    (0.00, 0.80), // gene_novelty_drive
    (0.50, 5.00), // gene_speed
];

/// Configuration for Darwinian evolution of agent genomes.
#[derive(Clone, Debug)]
pub struct SwarmEvolutionConfig {
    /// Enable per-agent genome evolution. When false, uses global PropagationConfig.
    pub enabled: bool,
    /// Health below which an agent is considered dead and eligible for replacement.
    pub death_threshold: f32,
    /// How often (in ticks) the reproduction pass runs.
    pub reproduction_interval: u32,
    /// Standard deviation of Gaussian mutation applied to each gene.
    pub mutation_sigma: f32,
    /// Health boost per tick for agents with surprise above reward threshold.
    pub health_reward: f32,
    /// Surprise level above which agents receive health reward.
    pub health_reward_threshold: f32,
}

impl Default for SwarmEvolutionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            death_threshold: 0.1,
            reproduction_interval: 50,
            mutation_sigma: 0.02,
            health_reward: 0.002,
            health_reward_threshold: 0.3,
        }
    }
}

/// The master orchestrator for the 100-Million Agent Swarm.
///
/// v4.0.0 Architecture:
/// - Memory-mapped SoA pool (OS-managed paging)
/// - Fibonacci spatial hash grid with real neighbor queries
/// - Pheromone-based stigmergic coordination
/// - SIRS surprise propagation with refractory dynamics
/// - Self-organized criticality via R₀ ≈ 1.0
pub struct SwarmEngineMaster {
    pub pool: MmapSwarmPool,
    pub pheromones: PheromoneField,
    pub grid: SpatialHashGrid,
    pub propagation_config: PropagationConfig,
    pub evolution_config: SwarmEvolutionConfig,

    // Config
    pub width: f32,
    pub height: f32,
    pub perception_radius: f32,
    pub global_tick: u64,
}

impl SwarmEngineMaster {
    pub fn new(n_agents: usize, width: f32, height: f32) -> Self {
        let mut pool = MmapSwarmPool::new(n_agents);
        pool.randomize_positions(width, height);

        // Scale pheromone field resolution based on agent count
        let pheromone_res = if n_agents >= 10_000_000 { 2048 } else { 1024 };
        let perception = 10.0; // agents perceive neighbors within 10 units

        Self {
            pool,
            pheromones: PheromoneField::new(
                pheromone_res,
                pheromone_res,
                width / pheromone_res as f32,
            ),
            grid: SpatialHashGrid::new(
                if n_agents >= 10_000_000 {
                    1 << 20
                } else {
                    1 << 18
                },
                perception, // cell_size = perception_radius for optimal 3x3 query
                [0.0, 0.0],
            ),
            propagation_config: PropagationConfig::default(),
            evolution_config: SwarmEvolutionConfig::default(),
            width,
            height,
            perception_radius: perception,
            global_tick: 0,
        }
    }

    /// The master tick function for 100M agents.
    ///
    /// Pipeline:
    /// 1. Spatial locality sorting (amortized every 100 ticks)
    /// 2. Rebuild spatial hash grid
    /// 3. Neighbor-driven physics: cohesion, separation, surprise propagation
    /// 4. Pheromone deposit + diffusion
    /// 5. Health decay
    pub fn tick(&mut self) {
        let start_time = Instant::now();
        self.global_tick += 1;

        // 1. Spatial locality sort (amortized O(N log N) every 100 ticks)
        if self.global_tick % 100 == 0 {
            self.pool.update_spatial_hashes(self.width);
            self.pool.sort_by_spatial_hash();
        }

        // 2. Rebuild spatial hash grid from current positions
        self.rebuild_grid();

        // 3. Real physics: neighbor queries drive agent behavior
        self.run_neighbor_physics();

        // 4. Agents deposit pheromones based on their surprise level
        self.deposit_agent_pheromones();

        // 5. Pheromone field diffusion + decay
        self.pheromones.tick();

        // 6. Health decay + evolution reward
        {
            let health = self.pool.health.as_mut_slice();
            let surprise = self.pool.surprise.as_slice();
            let evo = &self.evolution_config;
            for i in 0..self.pool.n_agents {
                health[i] *= 0.999;
                if evo.enabled && surprise[i] > evo.health_reward_threshold {
                    health[i] = (health[i] + evo.health_reward).min(1.0);
                }
            }
        }

        // 7. Evolution: death & reproduction
        if self.evolution_config.enabled
            && self.evolution_config.reproduction_interval > 0
            && self.global_tick % self.evolution_config.reproduction_interval as u64 == 0
        {
            self.run_evolution();
        }

        let elapsed = start_time.elapsed();
        if self.global_tick % 100 == 0 {
            println!(
                "Tick {}: {}M agents in {:?}",
                self.global_tick,
                self.pool.n_agents / 1_000_000,
                elapsed,
            );
        }
    }

    /// O(N) two-pass rebuild of the Fibonacci spatial hash grid.
    fn rebuild_grid(&mut self) {
        let n = self.pool.n_agents;
        let x = self.pool.x.as_slice();
        let y = self.pool.y.as_slice();

        // Pass 1: count agents per bucket
        self.grid.counts_reset();
        for i in 0..n {
            let (cx, cy) = self.grid.world_to_cell(x[i], y[i]);
            self.grid.count_agent(cx, cy);
        }

        // Prefix sum -> offsets
        self.grid.compute_offsets();

        // Pass 2: scatter agent indices into buckets
        for i in 0..n {
            let (cx, cy) = self.grid.world_to_cell(x[i], y[i]);
            self.grid.scatter_agent(cx, cy, i as u32);
        }
    }

    /// Real neighbor-driven physics.
    ///
    /// For each agent:
    /// 1. Query the spatial hash grid for neighbors within perception_radius
    /// 2. Compute cohesion force (move toward neighbor center-of-mass)
    /// 3. Compute separation force (avoid overlap with nearby agents)
    /// 4. Propagate surprise: if a neighbor has high surprise, absorb some
    /// 5. Follow pheromone trail gradients
    /// 6. Update velocity and position
    fn run_neighbor_physics(&mut self) {
        let n = self.pool.n_agents;
        let width = self.width;
        let height = self.height;
        let r = self.perception_radius;
        let r2 = r * r;
        let evo_enabled = self.evolution_config.enabled;

        // Scratch buffers for Jacobi-style synchronous update (no aliasing)
        let mut new_vx = vec![0.0f32; n];
        let mut new_vy = vec![0.0f32; n];
        let mut new_surprise = vec![0.0f32; n];
        let mut new_refractory = vec![0.0f32; n];

        // Read-only slices for current state
        let x = self.pool.x.as_slice();
        let y = self.pool.y.as_slice();
        let vx = self.pool.vx.as_slice();
        let vy = self.pool.vy.as_slice();
        let surprise = self.pool.surprise.as_slice();
        let refractory = self.pool.refractory.as_slice();

        // Per-agent genome slices (read when evolution enabled)
        let g_decay = self.pool.gene_decay.as_slice();
        let g_transfer = self.pool.gene_transfer.as_slice();
        let g_refractory = self.pool.gene_refractory.as_slice();
        let g_danger_sense = self.pool.gene_danger_sense.as_slice();
        let g_novelty_drive = self.pool.gene_novelty_drive.as_slice();
        let g_speed = self.pool.gene_speed.as_slice();

        // Propagation config (copy to avoid borrow conflicts with self)
        let cfg = self.propagation_config.clone();

        for i in 0..n {
            let px = x[i];
            let py = y[i];

            // Per-agent or global parameters
            let decay = if evo_enabled {
                g_decay[i]
            } else {
                cfg.surprise_decay
            };
            let transfer = if evo_enabled {
                g_transfer[i]
            } else {
                cfg.surprise_transfer
            };
            let refract_buildup = if evo_enabled {
                g_refractory[i]
            } else {
                cfg.refractory_buildup
            };
            let danger_fb = if evo_enabled {
                g_danger_sense[i]
            } else {
                cfg.danger_feedback
            };
            let novelty_attr = if evo_enabled {
                g_novelty_drive[i]
            } else {
                cfg.novelty_attraction
            };
            let max_speed = if evo_enabled { g_speed[i] } else { 2.0 };

            // Neighbor aggregation
            let mut neighbor_count = 0u32;
            let mut sum_x = 0.0f32;
            let mut sum_y = 0.0f32;
            let mut sep_x = 0.0f32;
            let mut sep_y = 0.0f32;

            // SIRS: distance-weighted mean of neighbor surprises
            let mut weighted_surprise_sum = 0.0f32;
            let mut weight_sum = 0.0f32;

            // Query the spatial hash grid for real neighbor candidates
            self.grid.query_neighbors(i as u32, px, py, r, |j| {
                let jx = x[j as usize];
                let jy = y[j as usize];

                let dx = jx - px;
                let dy = jy - py;
                let d2 = dx * dx + dy * dy;

                if d2 < r2 && d2 > 0.001 {
                    let dist = d2.sqrt();
                    neighbor_count += 1;

                    // Cohesion: accumulate neighbor center of mass
                    sum_x += jx;
                    sum_y += jy;

                    // Separation: repel from close neighbors
                    let inv = 1.0 / dist;
                    sep_x -= dx * inv;
                    sep_y -= dy * inv;

                    // Distance-weighted surprise: w = (1 - d/R)^falloff
                    let normalized_dist = dist / r;
                    let w = (1.0 - normalized_dist).max(0.0).powf(cfg.distance_falloff);
                    weighted_surprise_sum += surprise[j as usize] * w;
                    weight_sum += w;
                }
            });

            // --- Steering forces ---
            let mut fx = vx[i] * 0.9; // momentum
            let mut fy = vy[i] * 0.9;

            if neighbor_count > 0 {
                let nc = neighbor_count as f32;
                fx += (sum_x / nc - px) * 0.01; // cohesion
                fy += (sum_y / nc - py) * 0.01;
                fx += sep_x * 0.05; // separation
                fy += sep_y * 0.05;
            }

            // Pheromone gradient steering: trail + danger + novelty
            let (gx_trail, gy_trail) = self.pheromones.gradient(px, py, 2);
            let (gx_danger, gy_danger) = self.pheromones.gradient(px, py, 1);
            let (gx_novelty, gy_novelty) = self.pheromones.gradient(px, py, 4);
            fx += 0.3 * gx_trail - 0.5 * gx_danger + novelty_attr * gx_novelty;
            fy += 0.3 * gy_trail - 0.5 * gy_danger + novelty_attr * gy_novelty;

            // Random exploration
            fx += (rand::random::<f32>() - 0.5) * 0.1;
            fy += (rand::random::<f32>() - 0.5) * 0.1;

            // Clamp velocity (per-agent max speed)
            let mag = (fx * fx + fy * fy).sqrt().max(0.001);
            if mag > max_speed {
                fx = fx / mag * max_speed;
                fy = fy / mag * max_speed;
            }

            new_vx[i] = fx;
            new_vy[i] = fy;

            // --- SIRS Surprise Propagation (per-agent parameters) ---
            let mean_neighbor_surprise = if weight_sum > 0.0 {
                weighted_surprise_sum / weight_sum
            } else {
                0.0
            };

            let pheromone_input = self.pheromones.sample(px, py, 1) * danger_fb;
            let susceptibility = (1.0 - refractory[i]).max(0.0);
            let input = (mean_neighbor_surprise + pheromone_input) * transfer * susceptibility;
            let raw = (surprise[i] * decay + input).max(0.0);
            // Logistic saturation: approaches 1.0 but f32 rounds to 1.0 for raw > ~1e7.
            // Clamp to [0, 1) to prevent surprise=1.0 which violates SIRS model bounds.
            new_surprise[i] = (raw / (1.0 + raw)).min(1.0 - f32::EPSILON);

            // Refractory dynamics
            if surprise[i] > cfg.refractory_threshold {
                new_refractory[i] = (refractory[i] + refract_buildup).min(1.0);
            } else {
                new_refractory[i] = refractory[i] * cfg.refractory_decay;
            }
        }

        // Write back computed values
        let out_vx = self.pool.vx.as_mut_slice();
        let out_vy = self.pool.vy.as_mut_slice();
        let out_surprise = self.pool.surprise.as_mut_slice();
        let out_refractory = self.pool.refractory.as_mut_slice();
        let out_x = self.pool.x.as_mut_slice();
        let out_y = self.pool.y.as_mut_slice();

        for i in 0..n {
            out_vx[i] = new_vx[i];
            out_vy[i] = new_vy[i];
            out_surprise[i] = new_surprise[i];
            out_refractory[i] = new_refractory[i];
            out_x[i] = (out_x[i] + new_vx[i]).clamp(0.0, width);
            out_y[i] = (out_y[i] + new_vy[i]).clamp(0.0, height);
        }
    }

    /// Darwinian evolution: dead agents are replaced by mutated offspring of nearby healthy parents.
    ///
    /// For each dead agent (health < death_threshold):
    /// 1. Find a random healthy neighbor via the spatial hash grid
    /// 2. Copy the parent's genome with Gaussian mutation
    /// 3. Reset the child's state (health, surprise, refractory)
    fn run_evolution(&mut self) {
        let n = self.pool.n_agents;
        let evo = &self.evolution_config;
        let death_threshold = evo.death_threshold;
        let sigma = evo.mutation_sigma;
        let r = self.perception_radius;

        let health = self.pool.health.as_slice();
        let x = self.pool.x.as_slice();
        let y = self.pool.y.as_slice();

        // Collect dead agent indices
        let mut dead: Vec<usize> = Vec::new();
        for i in 0..n {
            if health[i] < death_threshold {
                dead.push(i);
            }
        }

        if dead.is_empty() {
            return;
        }

        // For each dead agent, find a healthy parent nearby and reproduce
        let mut rng = rand::thread_rng();

        // Read parent gene slices
        let parent_decay = self.pool.gene_decay.as_slice().to_vec();
        let parent_transfer = self.pool.gene_transfer.as_slice().to_vec();
        let parent_refractory = self.pool.gene_refractory.as_slice().to_vec();
        let parent_danger = self.pool.gene_danger_sense.as_slice().to_vec();
        let parent_novelty = self.pool.gene_novelty_drive.as_slice().to_vec();
        let parent_speed = self.pool.gene_speed.as_slice().to_vec();
        let parent_gen = self.pool.generation.as_slice().to_vec();
        let parent_health = health.to_vec();

        // Mutable slices for writing
        let out_gene_decay = self.pool.gene_decay.as_mut_slice();
        let out_gene_transfer = self.pool.gene_transfer.as_mut_slice();
        let out_gene_refractory = self.pool.gene_refractory.as_mut_slice();
        let out_gene_danger = self.pool.gene_danger_sense.as_mut_slice();
        let out_gene_novelty = self.pool.gene_novelty_drive.as_mut_slice();
        let out_gene_speed = self.pool.gene_speed.as_mut_slice();
        let out_generation = self.pool.generation.as_mut_slice();
        let out_health = self.pool.health.as_mut_slice();
        let out_surprise = self.pool.surprise.as_mut_slice();
        let out_refractory = self.pool.refractory.as_mut_slice();

        let mut births = 0u64;

        for &dead_idx in &dead {
            let px = x[dead_idx];
            let py = y[dead_idx];

            // Find a healthy parent among neighbors
            let mut parent_idx: Option<usize> = None;
            self.grid.query_neighbors(dead_idx as u32, px, py, r, |j| {
                let ji = j as usize;
                if parent_idx.is_none() && parent_health[ji] > 0.5 {
                    parent_idx = Some(ji);
                }
            });

            // If no nearby parent, try a random healthy agent
            if parent_idx.is_none() {
                for _ in 0..10 {
                    let candidate = rng.gen_range(0..n);
                    if parent_health[candidate] > 0.5 {
                        parent_idx = Some(candidate);
                        break;
                    }
                }
            }

            if let Some(pi) = parent_idx {
                // Gaussian mutation helper
                let mut mutate = |val: f32, min: f32, max: f32| -> f32 {
                    let noise = rng.gen::<f32>() * 2.0 - 1.0; // uniform [-1, 1] approximation
                    let noise2 = rng.gen::<f32>() * 2.0 - 1.0;
                    let gauss = noise + noise2; // rough Gaussian via sum of 2 uniforms
                    (val + gauss * sigma).clamp(min, max)
                };

                out_gene_decay[dead_idx] =
                    mutate(parent_decay[pi], GENE_RANGES[0].0, GENE_RANGES[0].1);
                out_gene_transfer[dead_idx] =
                    mutate(parent_transfer[pi], GENE_RANGES[1].0, GENE_RANGES[1].1);
                out_gene_refractory[dead_idx] =
                    mutate(parent_refractory[pi], GENE_RANGES[2].0, GENE_RANGES[2].1);
                out_gene_danger[dead_idx] =
                    mutate(parent_danger[pi], GENE_RANGES[3].0, GENE_RANGES[3].1);
                out_gene_novelty[dead_idx] =
                    mutate(parent_novelty[pi], GENE_RANGES[4].0, GENE_RANGES[4].1);
                out_gene_speed[dead_idx] =
                    mutate(parent_speed[pi], GENE_RANGES[5].0, GENE_RANGES[5].1);
                out_generation[dead_idx] = parent_gen[pi] + 1;

                // Reset child state
                out_health[dead_idx] = 1.0;
                out_surprise[dead_idx] = 0.0;
                out_refractory[dead_idx] = 0.0;

                births += 1;
            }
        }

        if self.global_tick % 100 == 0 && births > 0 {
            println!(
                "Evolution tick {}: {} deaths, {} births",
                self.global_tick,
                dead.len(),
                births
            );
        }
    }

    /// Agents deposit pheromones based on their state.
    /// High-surprise agents emit danger signals and novelty beacons (all agents checked).
    /// Trail markers use subsampled stride to avoid overwhelming the field at 100M scale.
    fn deposit_agent_pheromones(&mut self) {
        let x = self.pool.x.as_slice();
        let y = self.pool.y.as_slice();
        let surprise = self.pool.surprise.as_slice();
        let danger_threshold = self.propagation_config.danger_emission_threshold;
        let novelty_threshold = self.propagation_config.novelty_emission;
        let n = self.pool.n_agents;

        // Trail markers: subsampled (1 in 100) to avoid overwhelming the field
        let trail_stride = 100.max(1);
        for i in (0..n).step_by(trail_stride) {
            self.pheromones.deposit(x[i], y[i], 2, 0.1);
        }

        // Danger + novelty: every surprised agent deposits (critical for feedback loop)
        for i in 0..n {
            let s = surprise[i];
            if s > danger_threshold {
                self.pheromones.deposit(x[i], y[i], 1, s);
            }
            if s > novelty_threshold {
                self.pheromones.deposit(x[i], y[i], 4, s * 0.5);
            }
        }
    }
}
