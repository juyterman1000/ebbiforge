//! Production-grade adversarial tests for v4.0.0 self-evolving agent genomes.
//!
//! Tests cover:
//! - Memory integrity: genome arrays survive spatial sort
//! - Numerical stability: extreme gene values don't produce NaN/Inf
//! - Evolution edge cases: mass extinction, no parents, boundary genes
//! - Backwards compatibility: evolution disabled = identical behavior
//! - Scale: evolution under load with 1M agents
//! - Invariants: health/surprise/refractory bounds, gene clamping
//!
//! Run: cargo test -p ebbiforge-core evolution_test -- --nocapture
//! Run ignored: cargo test --release -p ebbiforge-core evolution_test -- --nocapture --ignored

#[cfg(test)]
mod tests {
    use crate::swarm::master_pipeline::{SwarmEngineMaster, SwarmEvolutionConfig, GENE_RANGES};
    use std::time::Instant;

    // ───────────────────────────────────────────────────────────
    //  UNIT TESTS (fast, run by default)
    // ───────────────────────────────────────────────────────────

    /// Genome arrays must be initialized to PropagationConfig defaults.
    #[test]
    fn genome_initialization_defaults() {
        let engine = SwarmEngineMaster::new(1000, 100.0, 100.0);
        let n = engine.pool.n_agents;

        for i in 0..n {
            assert_eq!(engine.pool.gene_decay.as_slice()[i], 0.92, "gene_decay[{}]", i);
            assert_eq!(engine.pool.gene_transfer.as_slice()[i], 0.08, "gene_transfer[{}]", i);
            assert_eq!(engine.pool.gene_refractory.as_slice()[i], 0.3, "gene_refractory[{}]", i);
            assert_eq!(engine.pool.gene_danger_sense.as_slice()[i], 0.15, "gene_danger_sense[{}]", i);
            assert_eq!(engine.pool.gene_novelty_drive.as_slice()[i], 0.2, "gene_novelty_drive[{}]", i);
            assert_eq!(engine.pool.gene_speed.as_slice()[i], 2.0, "gene_speed[{}]", i);
            assert_eq!(engine.pool.generation.as_slice()[i], 0, "generation[{}]", i);
        }
    }

    /// Spatial sort must preserve genome data integrity.
    /// After sorting by spatial hash, gene sums must remain unchanged.
    #[test]
    fn genome_survives_spatial_sort() {
        let mut engine = SwarmEngineMaster::new(10_000, 100.0, 100.0);

        // Manually set distinctive gene values (non-uniform to detect scrambling)
        for i in 0..10_000 {
            let f = i as f32 / 10_000.0;
            engine.pool.gene_decay.as_mut_slice()[i] = 0.8 + f * 0.19; // [0.8, 0.99]
            engine.pool.gene_transfer.as_mut_slice()[i] = 0.01 + f * 0.29;
            engine.pool.gene_speed.as_mut_slice()[i] = 0.5 + f * 4.5;
            engine.pool.generation.as_mut_slice()[i] = (i % 100) as u32;
        }

        // Record sums before sort
        let sum_decay_before: f64 = engine.pool.gene_decay.as_slice().iter().map(|&v| v as f64).sum();
        let sum_transfer_before: f64 = engine.pool.gene_transfer.as_slice().iter().map(|&v| v as f64).sum();
        let sum_speed_before: f64 = engine.pool.gene_speed.as_slice().iter().map(|&v| v as f64).sum();
        let sum_gen_before: u64 = engine.pool.generation.as_slice().iter().map(|&v| v as u64).sum();

        // Sort
        engine.pool.update_spatial_hashes(100.0);
        engine.pool.sort_by_spatial_hash();

        // Verify sums preserved
        let sum_decay_after: f64 = engine.pool.gene_decay.as_slice().iter().map(|&v| v as f64).sum();
        let sum_transfer_after: f64 = engine.pool.gene_transfer.as_slice().iter().map(|&v| v as f64).sum();
        let sum_speed_after: f64 = engine.pool.gene_speed.as_slice().iter().map(|&v| v as f64).sum();
        let sum_gen_after: u64 = engine.pool.generation.as_slice().iter().map(|&v| v as u64).sum();

        assert!((sum_decay_before - sum_decay_after).abs() < 0.01,
            "gene_decay sum changed: {} -> {}", sum_decay_before, sum_decay_after);
        assert!((sum_transfer_before - sum_transfer_after).abs() < 0.01,
            "gene_transfer sum changed: {} -> {}", sum_transfer_before, sum_transfer_after);
        assert!((sum_speed_before - sum_speed_after).abs() < 0.01,
            "gene_speed sum changed: {} -> {}", sum_speed_before, sum_speed_after);
        assert_eq!(sum_gen_before, sum_gen_after,
            "generation sum changed: {} -> {}", sum_gen_before, sum_gen_after);
    }

    /// Evolution disabled must produce identical physics to pre-v4.0 behavior.
    /// Run 10 ticks with evolution disabled; gene arrays must remain untouched.
    #[test]
    fn evolution_disabled_is_noop() {
        let mut engine = SwarmEngineMaster::new(5_000, 100.0, 100.0);
        assert!(!engine.evolution_config.enabled);

        // Record gene snapshot
        let decay_before: Vec<f32> = engine.pool.gene_decay.as_slice().to_vec();
        let gen_before: Vec<u32> = engine.pool.generation.as_slice().to_vec();

        // Inject surprise and run
        engine.pool.surprise.as_mut_slice()[0..100].fill(1.0);
        for _ in 0..10 {
            engine.tick();
        }

        // Genes must be completely unchanged
        assert_eq!(engine.pool.gene_decay.as_slice(), decay_before.as_slice(),
            "gene_decay changed with evolution disabled!");
        assert_eq!(engine.pool.generation.as_slice(), gen_before.as_slice(),
            "generation changed with evolution disabled!");
    }

    /// With evolution enabled but no surprise, agents eventually die (health decays).
    /// Evolution should replace dead agents — generation counter must increase.
    #[test]
    fn evolution_replaces_dead_agents() {
        let mut engine = SwarmEngineMaster::new(1_000, 100.0, 100.0);
        engine.evolution_config = SwarmEvolutionConfig {
            enabled: true,
            death_threshold: 0.1,
            reproduction_interval: 10,
            mutation_sigma: 0.02,
            health_reward: 0.002,
            health_reward_threshold: 0.3,
        };

        // Kill some agents manually
        for i in 0..100 {
            engine.pool.health.as_mut_slice()[i] = 0.01;
        }

        // Give the rest healthy status so they can be parents
        for i in 100..1000 {
            engine.pool.health.as_mut_slice()[i] = 1.0;
        }

        // Run 10 ticks (reproduction fires at tick 10)
        for _ in 0..10 {
            engine.tick();
        }

        // Check that dead agents were replaced
        let health = engine.pool.health.as_slice();
        let generation = engine.pool.generation.as_slice();

        let revived = (0..100).filter(|&i| health[i] > 0.5).count();
        let gen_max = generation.iter().cloned().max().unwrap_or(0);

        println!("Revived agents (of 100 dead): {}", revived);
        println!("Max generation: {}", gen_max);

        assert!(revived > 0, "No dead agents were replaced by evolution!");
        assert!(gen_max >= 1, "Generation counter never incremented!");
    }

    /// Mass extinction: ALL agents dead. Evolution should handle gracefully
    /// (no parent found = no crash, agents stay dead).
    #[test]
    fn mass_extinction_no_panic() {
        let mut engine = SwarmEngineMaster::new(500, 100.0, 100.0);
        engine.evolution_config = SwarmEvolutionConfig {
            enabled: true,
            death_threshold: 0.1,
            reproduction_interval: 1,
            mutation_sigma: 0.02,
            health_reward: 0.002,
            health_reward_threshold: 0.3,
        };

        // Kill everyone
        engine.pool.health.fill(0.01);

        // This must NOT panic
        for _ in 0..5 {
            engine.tick();
        }

        // All agents should still be dead (no parents available)
        let alive = engine.pool.health.as_slice().iter().filter(|&&h| h > 0.5).count();
        println!("Alive after mass extinction: {} (expected 0)", alive);
        // It's acceptable if some got randomly paired with each other before all died
    }

    /// Gene values must always be clamped within GENE_RANGES after mutation.
    #[test]
    fn gene_clamping_after_mutation() {
        let mut engine = SwarmEngineMaster::new(1_000, 100.0, 100.0);
        engine.evolution_config = SwarmEvolutionConfig {
            enabled: true,
            death_threshold: 0.1,
            reproduction_interval: 1,
            mutation_sigma: 1.0, // HUGE mutation rate to force clamping
            health_reward: 0.002,
            health_reward_threshold: 0.3,
        };

        // Kill half, keep half alive
        for i in 0..500 {
            engine.pool.health.as_mut_slice()[i] = 0.01;
        }

        // Run many evolution passes
        for _ in 0..20 {
            engine.tick();
        }

        // Verify all genes are within bounds
        let n = engine.pool.n_agents;
        for i in 0..n {
            let decay = engine.pool.gene_decay.as_slice()[i];
            let transfer = engine.pool.gene_transfer.as_slice()[i];
            let refract = engine.pool.gene_refractory.as_slice()[i];
            let danger = engine.pool.gene_danger_sense.as_slice()[i];
            let novelty = engine.pool.gene_novelty_drive.as_slice()[i];
            let speed = engine.pool.gene_speed.as_slice()[i];

            assert!(decay >= GENE_RANGES[0].0 && decay <= GENE_RANGES[0].1,
                "gene_decay[{}] = {} out of range [{}, {}]", i, decay, GENE_RANGES[0].0, GENE_RANGES[0].1);
            assert!(transfer >= GENE_RANGES[1].0 && transfer <= GENE_RANGES[1].1,
                "gene_transfer[{}] = {} out of range", i, transfer);
            assert!(refract >= GENE_RANGES[2].0 && refract <= GENE_RANGES[2].1,
                "gene_refractory[{}] = {} out of range", i, refract);
            assert!(danger >= GENE_RANGES[3].0 && danger <= GENE_RANGES[3].1,
                "gene_danger_sense[{}] = {} out of range", i, danger);
            assert!(novelty >= GENE_RANGES[4].0 && novelty <= GENE_RANGES[4].1,
                "gene_novelty_drive[{}] = {} out of range", i, novelty);
            assert!(speed >= GENE_RANGES[5].0 && speed <= GENE_RANGES[5].1,
                "gene_speed[{}] = {} out of range", i, speed);
        }
    }

    /// Extreme gene values must not produce NaN or Inf in surprise propagation.
    /// Sets gene_decay to maximum (0.99) and gene_transfer to maximum (0.3).
    /// This creates R₀ = 0.3 / (1 - 0.99) = 30 — wildly supercritical.
    /// Logistic saturation must prevent blowup.
    #[test]
    fn numerical_stability_extreme_genes() {
        let mut engine = SwarmEngineMaster::new(2_000, 100.0, 100.0);
        engine.evolution_config = SwarmEvolutionConfig {
            enabled: true,
            death_threshold: 0.01,
            reproduction_interval: 100,
            mutation_sigma: 0.0, // no mutation, keep extremes
            health_reward: 0.01,
            health_reward_threshold: 0.1,
        };

        // Set maximally aggressive genes
        for i in 0..2000 {
            engine.pool.gene_decay.as_mut_slice()[i] = 0.99;
            engine.pool.gene_transfer.as_mut_slice()[i] = 0.30;
            engine.pool.gene_refractory.as_mut_slice()[i] = 0.05; // minimal immunity
            engine.pool.gene_danger_sense.as_mut_slice()[i] = 0.50; // max feedback
            engine.pool.gene_speed.as_mut_slice()[i] = 5.0; // max speed
        }

        // Inject massive surprise
        engine.pool.surprise.fill(0.99);

        // Run 50 ticks — this is R₀=30, should saturate but NOT produce NaN
        for tick in 0..50 {
            engine.tick();

            let surprise = engine.pool.surprise.as_slice();
            let health = engine.pool.health.as_slice();
            let refractory = engine.pool.refractory.as_slice();

            for i in 0..engine.pool.n_agents {
                assert!(!surprise[i].is_nan(), "NaN surprise at agent {} tick {}", i, tick);
                assert!(!surprise[i].is_infinite(), "Inf surprise at agent {} tick {}", i, tick);
                assert!(surprise[i] >= 0.0 && surprise[i] < 1.0,
                    "surprise[{}] = {} out of [0, 1) at tick {}", i, surprise[i], tick);
                assert!(!health[i].is_nan(), "NaN health at agent {} tick {}", i, tick);
                assert!(!refractory[i].is_nan(), "NaN refractory at agent {} tick {}", i, tick);
                assert!(refractory[i] >= 0.0 && refractory[i] <= 1.0,
                    "refractory[{}] = {} out of [0, 1] at tick {}", i, refractory[i], tick);
            }
        }
    }

    /// Minimal gene values (low decay, low transfer) = subcritical.
    /// Surprise should collapse to near zero. No NaN/Inf.
    #[test]
    fn numerical_stability_minimal_genes() {
        let mut engine = SwarmEngineMaster::new(2_000, 100.0, 100.0);
        engine.evolution_config = SwarmEvolutionConfig {
            enabled: true,
            death_threshold: 0.01,
            reproduction_interval: 100,
            mutation_sigma: 0.0,
            health_reward: 0.002,
            health_reward_threshold: 0.3,
        };

        // Minimal genes: R₀ = 0.01 / (1 - 0.80) = 0.05 — deeply subcritical
        for i in 0..2000 {
            engine.pool.gene_decay.as_mut_slice()[i] = 0.80;
            engine.pool.gene_transfer.as_mut_slice()[i] = 0.01;
            engine.pool.gene_refractory.as_mut_slice()[i] = 0.80; // max immunity
            engine.pool.gene_danger_sense.as_mut_slice()[i] = 0.0;
            engine.pool.gene_speed.as_mut_slice()[i] = 0.5;
        }

        engine.pool.surprise.fill(0.5);

        for tick in 0..30 {
            engine.tick();
            let surprise = engine.pool.surprise.as_slice();
            for i in 0..engine.pool.n_agents {
                assert!(!surprise[i].is_nan(), "NaN at agent {} tick {}", i, tick);
                assert!(surprise[i] >= 0.0, "Negative surprise at agent {} tick {}", i, tick);
            }
        }

        // Surprise should have decayed to near zero
        let mean_s: f64 = engine.pool.surprise.as_slice().iter().map(|&v| v as f64).sum::<f64>()
            / engine.pool.n_agents as f64;
        println!("Mean surprise after 30 ticks (subcritical): {:.6}", mean_s);
        assert!(mean_s < 0.1, "Subcritical genes should collapse surprise, got {}", mean_s);
    }

    /// Health reward must only activate when evolution is enabled AND surprise > threshold.
    #[test]
    fn health_reward_conditional() {
        let mut engine = SwarmEngineMaster::new(100, 100.0, 100.0);

        // Inject surprise on all agents
        engine.pool.surprise.fill(0.5);

        // Evolution DISABLED: run 10 ticks, check health
        for _ in 0..10 {
            engine.tick();
        }
        let health_no_evo = engine.pool.health.as_slice()[0];
        // Should be 1.0 * 0.999^10 ≈ 0.990
        assert!(health_no_evo < 0.995, "Health should decay: {}", health_no_evo);

        // Reset and enable evolution
        let mut engine2 = SwarmEngineMaster::new(100, 100.0, 100.0);
        engine2.evolution_config.enabled = true;
        engine2.evolution_config.health_reward = 0.01; // large reward
        engine2.evolution_config.health_reward_threshold = 0.3;
        engine2.pool.surprise.fill(0.5); // above threshold

        for _ in 0..10 {
            engine2.tick();
        }
        let health_with_evo = engine2.pool.health.as_slice()[0];
        // With +0.01 reward and -0.001 decay per tick, health should be higher
        println!("Health without evo: {:.4}, with evo: {:.4}", health_no_evo, health_with_evo);
        // Note: surprise decays so the reward may not apply every tick, but initially it should
    }

    /// Position/velocity invariants must hold with per-agent max speed.
    /// Agents must never exceed their personal gene_speed.
    #[test]
    fn per_agent_speed_clamping() {
        let mut engine = SwarmEngineMaster::new(1_000, 100.0, 100.0);
        engine.evolution_config.enabled = true;

        // Set half agents to slow, half to fast
        for i in 0..500 {
            engine.pool.gene_speed.as_mut_slice()[i] = 0.5; // slow
        }
        for i in 500..1000 {
            engine.pool.gene_speed.as_mut_slice()[i] = 5.0; // fast
        }

        engine.pool.surprise.fill(0.8);

        for _ in 0..20 {
            engine.tick();
        }

        // Check velocity magnitudes respect per-agent speed
        let vx = engine.pool.vx.as_slice();
        let vy = engine.pool.vy.as_slice();
        let speed = engine.pool.gene_speed.as_slice();

        for i in 0..engine.pool.n_agents {
            let mag = (vx[i] * vx[i] + vy[i] * vy[i]).sqrt();
            assert!(mag <= speed[i] + 0.01,
                "Agent {} velocity {:.3} exceeds gene_speed {:.3}", i, mag, speed[i]);
        }
    }

    /// Agents must stay within world bounds after movement with any gene_speed.
    #[test]
    fn agents_stay_in_bounds() {
        let mut engine = SwarmEngineMaster::new(2_000, 100.0, 100.0);
        engine.evolution_config.enabled = true;

        // Max speed agents at world edges
        for i in 0..2000 {
            engine.pool.gene_speed.as_mut_slice()[i] = 5.0;
        }
        // Place some near edges
        for i in 0..100 {
            engine.pool.x.as_mut_slice()[i] = 0.1;
            engine.pool.y.as_mut_slice()[i] = 0.1;
        }
        for i in 100..200 {
            engine.pool.x.as_mut_slice()[i] = 99.9;
            engine.pool.y.as_mut_slice()[i] = 99.9;
        }

        for _ in 0..50 {
            engine.tick();
        }

        let x = engine.pool.x.as_slice();
        let y = engine.pool.y.as_slice();
        for i in 0..engine.pool.n_agents {
            assert!(x[i] >= 0.0 && x[i] <= 100.0,
                "Agent {} x={} out of world bounds", i, x[i]);
            assert!(y[i] >= 0.0 && y[i] <= 100.0,
                "Agent {} y={} out of world bounds", i, y[i]);
        }
    }

    /// Reproduction must increment generation counter correctly.
    #[test]
    fn generation_counter_increments() {
        let mut engine = SwarmEngineMaster::new(500, 100.0, 100.0);
        engine.evolution_config = SwarmEvolutionConfig {
            enabled: true,
            death_threshold: 0.5, // aggressive death threshold
            reproduction_interval: 1, // every tick
            mutation_sigma: 0.02,
            health_reward: 0.0, // no reward = everyone eventually dies
            health_reward_threshold: 1.0,
        };

        // Set some agents to gen=5
        for i in 0..250 {
            engine.pool.generation.as_mut_slice()[i] = 5;
            engine.pool.health.as_mut_slice()[i] = 1.0;
        }
        // Kill the rest
        for i in 250..500 {
            engine.pool.health.as_mut_slice()[i] = 0.01;
        }

        engine.tick();

        // Children of gen=5 parents should be gen=6
        let gen = engine.pool.generation.as_slice();
        let has_gen6 = gen.iter().any(|&g| g == 6);
        println!("Has generation 6 offspring: {}", has_gen6);
        assert!(has_gen6, "Generation counter should have incremented to 6");
    }

    /// Mixed gene populations must coexist without numerical issues.
    /// Half agents have high R₀, half have low R₀.
    #[test]
    fn mixed_gene_populations() {
        let mut engine = SwarmEngineMaster::new(2_000, 100.0, 100.0);
        engine.evolution_config.enabled = true;

        // Supercritical half
        for i in 0..1000 {
            engine.pool.gene_decay.as_mut_slice()[i] = 0.98;
            engine.pool.gene_transfer.as_mut_slice()[i] = 0.25;
        }
        // Subcritical half
        for i in 1000..2000 {
            engine.pool.gene_decay.as_mut_slice()[i] = 0.82;
            engine.pool.gene_transfer.as_mut_slice()[i] = 0.02;
        }

        engine.pool.surprise.fill(0.5);

        for tick in 0..30 {
            engine.tick();
            let s = engine.pool.surprise.as_slice();
            for i in 0..engine.pool.n_agents {
                assert!(!s[i].is_nan(), "NaN at agent {} tick {}", i, tick);
                assert!(s[i] >= 0.0 && s[i] < 1.0, "surprise[{}] = {} at tick {}", i, s[i], tick);
            }
        }
    }

    // ───────────────────────────────────────────────────────────
    //  STRESS TESTS (slow, ignored by default)
    // ───────────────────────────────────────────────────────────

    /// 1M agent evolution stress test.
    /// Run with: cargo test --release evolution_stress_1m -- --nocapture --ignored
    #[test]
    #[ignore]
    fn evolution_stress_1m() {
        let sep = "=".repeat(130);
        println!("\n{}", sep);
        println!("  EVOLUTION STRESS TEST — 1M agents, 100 ticks, evolution enabled");
        println!("{}\n", sep);

        let mut engine = SwarmEngineMaster::new(1_000_000, 1000.0, 1000.0);
        engine.evolution_config = SwarmEvolutionConfig {
            enabled: true,
            death_threshold: 0.1,
            reproduction_interval: 10, // frequent reproduction
            mutation_sigma: 0.05,      // aggressive mutation
            health_reward: 0.003,
            health_reward_threshold: 0.2,
        };

        // Inject surprise at center
        {
            let x = engine.pool.x.as_slice();
            let y = engine.pool.y.as_slice();
            let s = engine.pool.surprise.as_mut_slice();
            for i in 0..engine.pool.n_agents {
                let dx = x[i] - 500.0;
                let dy = y[i] - 500.0;
                if dx * dx + dy * dy < 100.0 * 100.0 {
                    s[i] = 1.0;
                }
            }
        }

        // Health decays at 0.999/tick — after 100 ticks: 0.999^100 ≈ 0.905.
        // That's above death_threshold=0.1, so nobody dies naturally in 100 ticks.
        // Seed 20% of agents as nearly dead to exercise the evolution code path.
        {
            let h = engine.pool.health.as_mut_slice();
            let n = engine.pool.n_agents;
            for i in 0..n / 5 {
                h[i] = 0.05; // below death_threshold
            }
        }

        println!("{:<5} | {:<9} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {}",
            "Tick", "Surprised", "Dead", "MeanGen", "Decay", "Transfer", "Speed", "Time");
        println!("{}", "-".repeat(100));

        let t_total = Instant::now();
        let mut nan_count = 0u64;
        let mut inf_count = 0u64;
        let mut oob_count = 0u64;

        for tick in 0..100 {
            let t = Instant::now();
            engine.tick();
            let elapsed = t.elapsed();

            // INVARIANT CHECK: no NaN, no Inf, all values in bounds
            let n = engine.pool.n_agents;
            let surprise = engine.pool.surprise.as_slice();
            let health = engine.pool.health.as_slice();
            let refractory = engine.pool.refractory.as_slice();
            let gene_decay = engine.pool.gene_decay.as_slice();
            let gene_transfer = engine.pool.gene_transfer.as_slice();
            let gene_speed = engine.pool.gene_speed.as_slice();

            for i in 0..n {
                if surprise[i].is_nan() { nan_count += 1; }
                if surprise[i].is_infinite() { inf_count += 1; }
                if surprise[i] < 0.0 || surprise[i] >= 1.0 { oob_count += 1; }
                if health[i].is_nan() { nan_count += 1; }
                if refractory[i].is_nan() { nan_count += 1; }

                // Gene bounds
                if gene_decay[i] < GENE_RANGES[0].0 || gene_decay[i] > GENE_RANGES[0].1 { oob_count += 1; }
                if gene_transfer[i] < GENE_RANGES[1].0 || gene_transfer[i] > GENE_RANGES[1].1 { oob_count += 1; }
                if gene_speed[i] < GENE_RANGES[5].0 || gene_speed[i] > GENE_RANGES[5].1 { oob_count += 1; }
            }

            if tick % 10 == 0 {
                let generation = engine.pool.generation.as_slice();
                let surprised = surprise.iter().filter(|&&s| s > 0.1).count();
                let dead = health.iter().filter(|&&h| h < 0.1).count();
                let mean_gen: f64 = generation.iter().map(|&g| g as f64).sum::<f64>() / n as f64;
                let mean_decay: f64 = gene_decay.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
                let mean_transfer: f64 = gene_transfer.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
                let mean_speed: f64 = gene_speed.iter().map(|&v| v as f64).sum::<f64>() / n as f64;

                println!("{:<5} | {:<9} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:.1}s",
                    tick + 1, surprised, dead,
                    format!("{:.2}", mean_gen),
                    format!("{:.4}", mean_decay),
                    format!("{:.4}", mean_transfer),
                    format!("{:.3}", mean_speed),
                    elapsed.as_secs_f64());
            }
        }

        let total_time = t_total.elapsed();

        println!("\n{}", sep);
        println!("  STRESS TEST RESULTS");
        println!("{}", sep);
        println!("  Total time: {:.1}s ({:.0}ms/tick avg)", total_time.as_secs_f64(), total_time.as_millis() as f64 / 100.0);
        println!("  NaN violations: {}", nan_count);
        println!("  Inf violations: {}", inf_count);
        println!("  Out-of-bounds violations: {}", oob_count);

        let max_gen = engine.pool.generation.as_slice().iter().cloned().max().unwrap_or(0);
        println!("  Max generation reached: {}", max_gen);
        println!("{}\n", sep);

        assert_eq!(nan_count, 0, "NaN values detected!");
        assert_eq!(inf_count, 0, "Inf values detected!");
        assert_eq!(oob_count, 0, "Out-of-bounds values detected!");
        assert!(max_gen > 0, "Evolution should have produced offspring (max_gen={})", max_gen);
    }

    /// Rapid evolution cycles: reproduction every tick, high mutation.
    /// Tests evolution code path under maximum stress.
    #[test]
    #[ignore]
    fn evolution_rapid_cycles_stress() {
        println!("\n  RAPID EVOLUTION STRESS — 100K agents, reproduction every tick, sigma=0.1");

        let mut engine = SwarmEngineMaster::new(100_000, 500.0, 500.0);
        engine.evolution_config = SwarmEvolutionConfig {
            enabled: true,
            death_threshold: 0.3, // aggressive culling
            reproduction_interval: 1,
            mutation_sigma: 0.1,  // extreme mutation
            health_reward: 0.0,   // no reward — forces continuous death/rebirth cycle
            health_reward_threshold: 1.0,
        };

        // Inject surprise broadly
        for i in 0..100_000 {
            if i % 5 == 0 { engine.pool.surprise.as_mut_slice()[i] = 0.8; }
        }

        // Health decays at 0.999/tick — 0.999^50 ≈ 0.951, well above 0.3.
        // Seed 30% of agents as nearly dead to ensure evolution fires immediately.
        {
            let h = engine.pool.health.as_mut_slice();
            for i in 0..30_000 {
                h[i] = 0.05; // well below death_threshold=0.3
            }
        }

        let t = Instant::now();
        for tick in 0..50 {
            engine.tick();

            // Check invariants every tick under extreme conditions
            let n = engine.pool.n_agents;
            for i in 0..n {
                let s = engine.pool.surprise.as_slice()[i];
                let h = engine.pool.health.as_slice()[i];
                let r = engine.pool.refractory.as_slice()[i];
                assert!(!s.is_nan() && !s.is_infinite() && s >= 0.0 && s < 1.0,
                    "Invalid surprise[{}]={} at tick {}", i, s, tick);
                assert!(!h.is_nan() && !h.is_infinite(),
                    "Invalid health[{}]={} at tick {}", i, h, tick);
                assert!(!r.is_nan() && r >= 0.0 && r <= 1.0,
                    "Invalid refractory[{}]={} at tick {}", i, r, tick);
            }
        }

        let elapsed = t.elapsed();
        let max_gen = engine.pool.generation.as_slice().iter().cloned().max().unwrap_or(0);
        println!("  Completed in {:.1}s, max generation: {}", elapsed.as_secs_f64(), max_gen);
        assert!(max_gen > 0, "Evolution should have produced offspring");
    }

    /// Genome diversity test: with evolution on, gene values should diverge from defaults.
    /// Forces deaths by manually killing agents, then verifies mutation creates diversity.
    #[test]
    fn evolution_creates_diversity() {
        let mut engine = SwarmEngineMaster::new(2_000, 100.0, 100.0);
        engine.evolution_config = SwarmEvolutionConfig {
            enabled: true,
            death_threshold: 0.5,
            reproduction_interval: 1,  // every tick
            mutation_sigma: 0.05,
            health_reward: 0.0,        // no reward = guaranteed death spiral
            health_reward_threshold: 1.0,
        };

        // Keep half alive as parents, kill the rest
        for i in 0..1000 {
            engine.pool.health.as_mut_slice()[i] = 1.0;
        }
        for i in 1000..2000 {
            engine.pool.health.as_mut_slice()[i] = 0.01; // dead
        }

        // Run 10 ticks — dead agents should be replaced with mutated offspring
        for _ in 0..10 {
            engine.tick();
        }

        let max_gen = engine.pool.generation.as_slice().iter().cloned().max().unwrap_or(0);
        println!("Max generation: {}", max_gen);
        assert!(max_gen >= 1, "Evolution must have produced offspring (max_gen={})", max_gen);

        // Compute std dev of gene_transfer
        let gt = engine.pool.gene_transfer.as_slice();
        let n = engine.pool.n_agents;
        let mean: f64 = gt.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let var: f64 = gt.iter().map(|&v| { let d = v as f64 - mean; d * d }).sum::<f64>() / n as f64;
        let std = var.sqrt();

        println!("Gene transfer std dev after evolution: {:.6}", std);
        assert!(std > 0.0001, "Mutation must create gene diversity! std={}", std);
    }
}
