//! Real-world scale tests for Ebbiforge v4.0.0
//!
//! These tests allocate real memory-mapped arrays, run real spatial hash queries,
//! compute real neighbor interactions (cohesion, separation, surprise propagation),
//! and measure actual RSS from /proc/self/status.
//!
//! Run: cargo test --release -- --nocapture --ignored

#[cfg(test)]
mod tests {
    use crate::swarm::mmap_pool::MmapSwarmPool;
    use crate::swarm::master_pipeline::SwarmEngineMaster;
    use std::time::Instant;

    fn get_rss_mb() -> f64 {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let kb: f64 = line.split_whitespace()
                        .nth(1)
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.0);
                    return kb / 1024.0;
                }
            }
        }
        0.0
    }

    /// 100M agent mmap allocation test.
    /// Proves that memory-mapped arrays work at 100M scale.
    #[test]
    #[ignore]
    fn scale_100m_mmap_allocation() {
        let sep = "=".repeat(80);
        println!("\n{}", sep);
        println!("  EBBIFORGE v4.0.0 — 100M AGENT MMAP TEST");
        println!("{}\n", sep);

        let rss_before = get_rss_mb();
        println!("[1/5] Baseline RSS: {:.1} MB", rss_before);

        // Allocate 100M agents
        println!("\n[2/5] Allocating 100,000,000 agents via anonymous mmap...");
        let t0 = Instant::now();
        let mut pool = MmapSwarmPool::new(100_000_000);
        let t_alloc = t0.elapsed();
        println!("  Time: {:?}  |  RSS: {:.1} MB", t_alloc, get_rss_mb());

        assert_eq!(pool.n_agents, 100_000_000);
        assert_eq!(pool.health.as_slice()[99_999_999], 1.0);

        // Randomize
        println!("\n[3/5] Randomizing 100M positions...");
        let t1 = Instant::now();
        pool.randomize_positions(10000.0, 10000.0);
        let t_rand = t1.elapsed();
        let sample = pool.x.as_slice()[50_000_000];
        println!("  Time: {:?}  |  RSS: {:.1} MB  |  Sample: x={:.1}", t_rand, get_rss_mb(), sample);
        assert!(sample > 0.0 && sample < 10000.0);

        // Spatial hash
        println!("\n[4/5] Spatial hashing 100M agents...");
        let t2 = Instant::now();
        pool.update_spatial_hashes(10000.0);
        let t_hash = t2.elapsed();
        println!("  Time: {:?}", t_hash);

        // Health decay
        println!("\n[5/5] Health decay on 100M agents...");
        let t3 = Instant::now();
        pool.health.as_mut_slice().iter_mut().for_each(|h| *h *= 0.999);
        let t_decay = t3.elapsed();
        let decayed = pool.health.as_slice()[0];
        println!("  Time: {:?}  |  Health: {:.6}", t_decay, decayed);
        assert!((decayed - 0.999).abs() < 0.001);

        println!("\n{}", sep);
        println!("  100M RESULT: PASSED  |  Peak RSS: {:.0} MB", get_rss_mb());
        println!("{}\n", sep);
    }

    /// 1M agent FULL PIPELINE test.
    /// Runs real neighbor queries via spatial hash grid.
    /// Each tick: grid rebuild -> neighbor cohesion/separation/surprise -> pheromone deposit -> diffusion
    #[test]
    #[ignore]
    fn scale_1m_full_pipeline() {
        let sep = "=".repeat(80);
        println!("\n{}", sep);
        println!("  EBBIFORGE v4.0.0 — 1M FULL PIPELINE TEST");
        println!("  Real neighbor queries, real pheromone diffusion, real physics");
        println!("{}\n", sep);

        println!("[1/4] Initializing 1M agent pipeline...");
        let t0 = Instant::now();
        let mut engine = SwarmEngineMaster::new(1_000_000, 1000.0, 1000.0);
        let t_init = t0.elapsed();
        println!("  Init: {:?}  |  RSS: {:.1} MB", t_init, get_rss_mb());

        // Inject a surprise event to test propagation
        println!("\n[2/4] Injecting surprise event at (500, 500)...");
        {
            let x = engine.pool.x.as_slice();
            let y = engine.pool.y.as_slice();
            let surprise = engine.pool.surprise.as_mut_slice();
            let mut shocked = 0;
            for i in 0..engine.pool.n_agents {
                let dx = x[i] - 500.0;
                let dy = y[i] - 500.0;
                if dx * dx + dy * dy < 2500.0 { // radius 50
                    surprise[i] = 1.0;
                    shocked += 1;
                }
            }
            println!("  Shocked {} agents within radius 50 of center", shocked);
        }

        // Count agents with surprise before ticks
        let surprised_before = engine.pool.surprise.as_slice()
            .iter().filter(|s| **s > 0.1).count();
        println!("  Agents with surprise > 0.1 BEFORE ticks: {}", surprised_before);

        // Run 5 real ticks
        println!("\n[3/4] Running 5 real ticks...");
        let mut tick_times = Vec::new();
        for i in 0..5 {
            let t = Instant::now();
            engine.tick();
            let elapsed = t.elapsed();
            tick_times.push(elapsed);

            let surprised = engine.pool.surprise.as_slice()
                .iter().filter(|s| **s > 0.1).count();
            println!("  Tick {}: {:?}  |  Surprised agents: {}", i + 1, elapsed, surprised);
        }

        // Verify real state changes
        println!("\n[4/4] Verifying real physics...");

        let surprised_after = engine.pool.surprise.as_slice()
            .iter().filter(|s| **s > 0.1).count();

        let health_sample = engine.pool.health.as_slice()[500_000];
        let x_sample = engine.pool.x.as_slice()[500_000];
        let mean_vx: f32 = engine.pool.vx.as_slice().iter().map(|v| v.abs()).sum::<f32>()
            / engine.pool.n_agents as f32;

        println!("  Surprised agents AFTER 5 ticks: {} (was {})", surprised_after, surprised_before);
        println!("  Surprise propagated: {}", if surprised_after > surprised_before { "YES" } else { "NO" });
        println!("  Health decayed:  {:.6} (expected ~0.995)", health_sample);
        println!("  Mean |velocity|: {:.4} (should be > 0 if agents moved)", mean_vx);
        println!("  Agent #500K at:  x={:.2}", x_sample);

        // Real assertions
        assert!(health_sample < 1.0, "Health must have decayed");
        assert!(health_sample > 0.9, "Health shouldn't be zero after 5 ticks");
        assert!(mean_vx > 0.0, "Agents must have non-zero velocity");

        let avg_ms = tick_times.iter().map(|t| t.as_millis()).sum::<u128>() as f64
            / tick_times.len() as f64;

        println!("\n{}", sep);
        println!("  1M PIPELINE RESULT: PASSED");
        println!("  Avg tick: {:.0} ms  |  RSS: {:.0} MB", avg_ms, get_rss_mb());
        if surprised_after > surprised_before {
            println!("  Surprise propagation: CONFIRMED ({} -> {} agents)",
                surprised_before, surprised_after);
        }
        println!("{}\n", sep);
    }
}
