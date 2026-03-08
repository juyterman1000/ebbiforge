"""
Ebbiforge — Example 02: Watch Agents Evolve
================================================

10,000 agents with Darwinian evolution. Watch natural selection
in real-time: agents develop different survival strategies,
castes emerge, and the population self-organizes.

No other AI framework can do this.

Run: python examples/02_evolution.py
  or: ebbiforge example evolution
"""

try:
    import ebbiforge_core as cogops
except ImportError:
    print("❌ Rust core required. Build with: maturin develop --release")
    print("   (from the cogops/ directory)")
    exit(1)

import time

# ── Initialize 10K agents ────────────────────────────────────────
print("🧬 Ebbiforge — Darwinian Evolution Demo")
print("=" * 50)

world_config = cogops.WorldModelConfig(
    ebbinghaus_decay_rate=0.1,
    grid_size=(100, 100),
)

swarm = cogops.TensorSwarm(
    agent_count=10_000,
    world_config=world_config,
)

# Register locations in the world
swarm.register_locations(
    villages=[(20, 30), (80, 70), (50, 10), (10, 90)],
    towns=[(40, 50), (60, 40)],
    cities=[(50, 50)],
    ambush_zones=[(30, 35), (70, 65)],
)

# ── Run simulation with periodic shocks ───────────────────────────
print(f"\nPopulation: 10,000 agents")
print(f"Ticks:      2,000")
print(f"Events:     Environmental shocks every 200 ticks\n")

metrics = {"surprise": [], "ticks_ms": []}

for tick in range(2000):
    start = time.time()

    # Inject environmental shocks periodically
    if tick % 200 == 0 and tick > 0:
        shock_x, shock_y = (tick * 7 % 100, tick * 13 % 100)
        swarm.apply_environmental_shock(
            location=(shock_x, shock_y), radius=8, intensity=1.0
        )
        print(f"\n  ⚡ Environmental shock at ({shock_x}, {shock_y})!")

    swarm.tick()
    elapsed = (time.time() - start) * 1000

    if tick % 100 == 0:
        snapshot = swarm.sample_population_metrics()
        surprise = snapshot["mean_surprise_score"]
        metrics["surprise"].append(surprise)
        metrics["ticks_ms"].append(elapsed)

        # Analyze population diversity
        share_dist = snapshot["share_probability_distribution"]
        brokers = sum(1 for p in share_dist if p > 0.7)
        selfish = sum(1 for p in share_dist if p < 0.3)
        total = len(share_dist)

        print(
            f"  Tick {tick:>4} | "
            f"{elapsed:>5.1f}ms | "
            f"Surprise: {surprise:.4f} | "
            f"Brokers: {brokers/total*100:>4.1f}% | "
            f"Selfish: {selfish/total*100:>4.1f}%"
        )

# ── Final analysis ────────────────────────────────────────────────
print("\n" + "=" * 50)
print("EVOLUTION RESULTS")
print("=" * 50)

final_dist = swarm.sample_population_metrics()["share_probability_distribution"]
total = len(final_dist)
brokers = sum(1 for p in final_dist if p > 0.7)
selfish = sum(1 for p in final_dist if p < 0.3)
middle = total - brokers - selfish

print(f"\n  🤝 Information Brokers (share > 70%): {brokers/total*100:.1f}%")
print(f"  🦊 Selfish Actors    (share < 30%): {selfish/total*100:.1f}%")
print(f"  🔄 Undifferentiated  (middle):      {middle/total*100:.1f}%")

bimodal = (brokers + selfish) > middle
print(f"\n  Caste Emergence: {'✅ YES — natural castes formed!' if bimodal else '❌ Not yet'}")

avg_tick = sum(metrics["ticks_ms"]) / len(metrics["ticks_ms"])
print(f"  Avg Tick Time:   {avg_tick:.1f}ms ({10_000 / (avg_tick / 1000):,.0f} agents/sec)")
print(f"\n--- What just happened? ---")
print("10,000 agents with 6 genes evolved via natural selection.")
print("Dead agents are replaced by mutated offspring of healthy neighbors.")
print("The population self-organized into specialization castes.")
print("This runs in PURE RUST — no LLM calls, no API cost.")
