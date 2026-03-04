//! Criticality analysis: Does surprise cascade, stabilize, or collapse?
//!
//! SIRS model with refractory dynamics and pheromone feedback.
//! Run: cargo test --release criticality_analysis -- --nocapture --ignored

#[cfg(test)]
mod tests {
    use crate::swarm::master_pipeline::SwarmEngineMaster;
    use std::time::Instant;

    /// SIRS criticality analysis with R₀ tracking.
    ///
    /// Injects a surprise event at the center, then tracks:
    /// - Surprised agent count and delta
    /// - R₀_eff (effective reproduction number, should oscillate around 1.0)
    /// - Mean refractory level (negative feedback indicator)
    /// - Spatial spread from injection origin
    /// - Danger pheromone density (feedback loop health)
    #[test]
    #[ignore]
    fn criticality_analysis() {
        let sep = "=".repeat(120);
        println!("\n{}", sep);
        println!("  SIRS CRITICALITY ANALYSIS — Surprise Propagation with Refractory Dynamics");
        println!("  1M agents, injection at (500,500), radius 50");
        println!("  R₀_base = {:.3}", 0.08f32 / (1.0 - 0.92));
        println!("{}\n", sep);

        let mut engine = SwarmEngineMaster::new(1_000_000, 1000.0, 1000.0);

        // Inject surprise at center
        let origin = (500.0f32, 500.0f32);
        let inject_radius = 50.0f32;
        {
            let x = engine.pool.x.as_slice();
            let y = engine.pool.y.as_slice();
            let surprise = engine.pool.surprise.as_mut_slice();
            for i in 0..engine.pool.n_agents {
                let dx = x[i] - origin.0;
                let dy = y[i] - origin.1;
                if dx * dx + dy * dy < inject_radius * inject_radius {
                    surprise[i] = 1.0;
                }
            }
        }

        println!(
            "{:<5} | {:<9} | {:<8} | {:<7} | {:<7} | {:<7} | {:<9} | {:<7} | {:<7} | {}",
            "Tick",
            "Surprised",
            "Delta",
            "% Pop",
            "R₀_eff",
            "Mean R",
            "Mean Dist",
            "Peak S",
            "Danger",
            "Time"
        );
        println!("{}", "-".repeat(120));

        let mut prev_count = 0u64;
        let mut _prev_delta = 0i64;

        for tick in 0..30 {
            let t = Instant::now();

            let state = measure_surprise_state(&engine, origin);

            if tick == 0 {
                prev_count = state.count;
                println!(
                    "{:<5} | {:<9} | {:<8} | {:<7} | {:<7} | {:<7} | {:<9} | {:<7} | {:<7} | {}",
                    "init",
                    state.count,
                    "-",
                    format!("{:.2}%", state.count as f64 / 10000.0),
                    format!("{:.3}", state.r0_eff),
                    format!("{:.4}", state.mean_refractory),
                    format!("{:.1}", state.mean_dist),
                    format!("{:.3}", state.peak),
                    format!("{:.4}", state.danger_density),
                    "-"
                );
            }

            engine.tick();
            let elapsed = t.elapsed();

            let new_state = measure_surprise_state(&engine, origin);
            let delta = new_state.count as i64 - prev_count as i64;

            println!(
                "{:<5} | {:<9} | {:<8} | {:<7} | {:<7} | {:<7} | {:<9} | {:<7} | {:<7} | {:.1}s",
                tick + 1,
                new_state.count,
                format!("{:+}", delta),
                format!("{:.2}%", new_state.count as f64 / 10000.0),
                format!("{:.3}", new_state.r0_eff),
                format!("{:.4}", new_state.mean_refractory),
                format!("{:.1}", new_state.mean_dist),
                format!("{:.4}", new_state.peak),
                format!("{:.4}", new_state.danger_density),
                elapsed.as_secs_f64()
            );

            _prev_delta = delta;
            prev_count = new_state.count;
        }

        let final_state = measure_surprise_state(&engine, origin);

        println!("\n{}", sep);
        println!("  SIRS ANALYSIS");
        println!("{}", sep);

        let r0 = final_state.r0_eff;
        let final_pct = final_state.count as f64 / 10000.0;

        if r0 > 1.2 {
            println!("  VERDICT: SUPERCRITICAL (R₀_eff = {:.3})", r0);
            println!(
                "  Refractory feedback insufficient — runaway propagation at {:.1}%",
                final_pct
            );
        } else if r0 > 0.8 && final_state.count > 0 && final_state.peak > 0.05 {
            println!("  VERDICT: CRITICAL — SELF-ORGANIZED (R₀_eff = {:.3})", r0);
            println!(
                "  Surprise wave sustained at {:.1}% via refractory negative feedback",
                final_pct
            );
            println!(
                "  Mean refractory: {:.4} (immune population fraction)",
                final_state.mean_refractory
            );
        } else {
            println!("  VERDICT: SUBCRITICAL (R₀_eff = {:.3})", r0);
            println!("  Surprise collapsed — wave could not sustain propagation",);
        }

        println!("  R₀_base: {:.3}", engine.propagation_config.r0_base());
        println!("  R₀_eff:  {:.3} (= R₀_base × (1 - mean_refractory))", r0);
        println!(
            "  Danger pheromone density: {:.4}",
            final_state.danger_density
        );
        println!(
            "  Final mean distance from origin: {:.1} units (σ={:.1})",
            final_state.mean_dist, final_state.std_dev
        );
        println!("  Peak surprise remaining: {:.4}", final_state.peak);
        println!("{}\n", sep);
    }

    /// Evolution analysis: tracks gene drift under Darwinian selection.
    ///
    /// Run: cargo test --release -p ebbiforge-core evolution_analysis -- --nocapture --ignored
    #[test]
    #[ignore]
    fn evolution_analysis() {
        use crate::swarm::master_pipeline::SwarmEvolutionConfig;

        let sep = "=".repeat(130);
        println!("\n{}", sep);
        println!("  EVOLUTION ANALYSIS — Self-Evolving Agent Genomes (v4.0.0)");
        println!("  1M agents, evolution enabled, 200 ticks");
        println!("{}\n", sep);

        let mut engine = SwarmEngineMaster::new(1_000_000, 1000.0, 1000.0);
        engine.evolution_config = SwarmEvolutionConfig {
            enabled: true,
            death_threshold: 0.1,
            reproduction_interval: 50,
            mutation_sigma: 0.02,
            health_reward: 0.002,
            health_reward_threshold: 0.3,
        };

        // Inject surprise at center
        {
            let x = engine.pool.x.as_slice();
            let y = engine.pool.y.as_slice();
            let surprise = engine.pool.surprise.as_mut_slice();
            for i in 0..engine.pool.n_agents {
                let dx = x[i] - 500.0;
                let dy = y[i] - 500.0;
                if dx * dx + dy * dy < 50.0 * 50.0 {
                    surprise[i] = 1.0;
                }
            }
        }

        println!(
            "{:<5} | {:<9} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {}",
            "Tick",
            "Surprised",
            "Deaths",
            "MeanGen",
            "Decay",
            "Transfer",
            "Refract",
            "Danger",
            "Speed",
            "Time"
        );
        println!("{}", "-".repeat(130));

        for tick in 0..200 {
            let t = Instant::now();
            engine.tick();
            let elapsed = t.elapsed();

            if tick % 10 == 0 {
                let n = engine.pool.n_agents;
                let surprise = engine.pool.surprise.as_slice();
                let health = engine.pool.health.as_slice();
                let gene_decay = engine.pool.gene_decay.as_slice();
                let gene_transfer = engine.pool.gene_transfer.as_slice();
                let gene_refractory = engine.pool.gene_refractory.as_slice();
                let gene_danger = engine.pool.gene_danger_sense.as_slice();
                let gene_speed = engine.pool.gene_speed.as_slice();
                let generation = engine.pool.generation.as_slice();

                let surprised = surprise.iter().filter(|&&s| s > 0.1).count();
                let dead = health.iter().filter(|&&h| h < 0.1).count();
                let mean_gen: f64 = generation.iter().map(|&g| g as f64).sum::<f64>() / n as f64;
                let mean_decay: f64 = gene_decay.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
                let mean_transfer: f64 =
                    gene_transfer.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
                let mean_refract: f64 =
                    gene_refractory.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
                let mean_danger: f64 =
                    gene_danger.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
                let mean_speed: f64 = gene_speed.iter().map(|&v| v as f64).sum::<f64>() / n as f64;

                println!("{:<5} | {:<9} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:.1}s",
                    tick + 1, surprised, dead,
                    format!("{:.2}", mean_gen),
                    format!("{:.4}", mean_decay),
                    format!("{:.4}", mean_transfer),
                    format!("{:.4}", mean_refract),
                    format!("{:.4}", mean_danger),
                    format!("{:.3}", mean_speed),
                    elapsed.as_secs_f64());
            }
        }

        // Final gene diversity analysis
        let n = engine.pool.n_agents;
        let gene_transfer = engine.pool.gene_transfer.as_slice();
        let generation = engine.pool.generation.as_slice();

        let mean_t: f64 = gene_transfer.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let var_t: f64 = gene_transfer
            .iter()
            .map(|&v| {
                let d = v as f64 - mean_t;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let std_t = var_t.sqrt();

        let max_gen = generation.iter().cloned().max().unwrap_or(0);
        let mean_gen: f64 = generation.iter().map(|&g| g as f64).sum::<f64>() / n as f64;

        println!("\n{}", sep);
        println!("  EVOLUTION SUMMARY");
        println!("{}", sep);
        println!("  Mean gene_transfer: {:.4} (started at 0.08)", mean_t);
        println!("  Std dev gene_transfer: {:.4} (diversity)", std_t);
        println!("  Max generation: {}", max_gen);
        println!("  Mean generation: {:.2}", mean_gen);
        println!("{}\n", sep);
    }

    struct SurpriseState {
        count: u64,
        mean_dist: f64,
        peak: f32,
        std_dev: f64,
        mean_refractory: f64,
        r0_eff: f32,
        danger_density: f64,
    }

    fn measure_surprise_state(engine: &SwarmEngineMaster, origin: (f32, f32)) -> SurpriseState {
        let x = engine.pool.x.as_slice();
        let y = engine.pool.y.as_slice();
        let surprise = engine.pool.surprise.as_slice();
        let refractory = engine.pool.refractory.as_slice();
        let n = engine.pool.n_agents;

        let threshold = 0.1;
        let mut count = 0u64;
        let mut sum_dist = 0.0f64;
        let mut peak = 0.0f32;
        let mut sum_refractory = 0.0f64;

        for i in 0..n {
            sum_refractory += refractory[i] as f64;
            if surprise[i] > threshold {
                count += 1;
                let dx = (x[i] - origin.0) as f64;
                let dy = (y[i] - origin.1) as f64;
                sum_dist += (dx * dx + dy * dy).sqrt();
                if surprise[i] > peak {
                    peak = surprise[i];
                }
            }
        }

        let mean_dist = if count > 0 {
            sum_dist / count as f64
        } else {
            0.0
        };
        let mean_refractory = sum_refractory / n as f64;

        // Spatial std dev
        let mut sum_sq = 0.0f64;
        if count > 0 {
            for i in 0..n {
                if surprise[i] > threshold {
                    let dx = (x[i] - origin.0) as f64;
                    let dy = (y[i] - origin.1) as f64;
                    let d = (dx * dx + dy * dy).sqrt();
                    sum_sq += (d - mean_dist) * (d - mean_dist);
                }
            }
        }
        let std_dev = if count > 1 {
            (sum_sq / (count - 1) as f64).sqrt()
        } else {
            0.0
        };

        // Danger pheromone: sample mean density across grid
        let danger_density = {
            let field = &engine.pheromones;
            let ch_off = 1 * field.width * field.height;
            let ch_slice = &field.data[ch_off..ch_off + field.width * field.height];
            let sum: f64 = ch_slice.iter().map(|v| *v as f64).sum();
            sum / ch_slice.len() as f64
        };

        let r0_eff = engine
            .propagation_config
            .r0_effective(mean_refractory as f32);

        SurpriseState {
            count,
            mean_dist,
            peak,
            std_dev,
            mean_refractory,
            r0_eff,
            danger_density,
        }
    }
}
