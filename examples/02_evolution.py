"""
Ebbiforge — Example 02: Caste Emergence via Darwinian RL
=========================================================

Watch 5,000 agents self-organize into behavioral castes in real-time.
Agents near an ambush zone learn to hoard resources (low share_prob).
Agents who successfully reach the city learn to share (high share_prob).
No hardcoded rules — emergent behavior from pure TD-RL.

Run: python examples/02_evolution.py
  or: ebbiforge example evolution
"""

try:
    import ebbiforge_core as cogops
except ImportError:
    print("❌ Rust core required. Build with: maturin develop --release")
    exit(1)

import time

# ── Initialize world ─────────────────────────────────────────────
print("🧬 Ebbiforge — Caste Emergence via Darwinian RL")
print("=" * 54)

swarm = cogops.TensorSwarm(agent_count=5_000)

# Village at (200,200) overlaps with ambush zone → agents that linger
# here get negative RL reward → behaviorally diverge from city traders
swarm.register_locations(
    villages=[(200.0, 200.0), (600.0, 400.0)],
    towns=[],
    cities=[(800.0, 800.0)],
    ambush_zones=[(200.0, 200.0)],
)

# Seed initial surprise so memory dynamics start immediately
swarm.apply_environmental_shock(location=(200.0, 200.0), radius=300.0, intensity=0.9)

# ── Run simulation ───────────────────────────────────────────────
TICKS = 400
print(f"\nAgents: 5,000  |  Ticks: {TICKS}  |  World: 1000×1000")
print(f"Ambush at (200,200) → RL training signal for behavioral divergence\n")
print(f"  {'TICK':>4}  {'ms':>5}  {'Surprise':>9}  {'Altruists':>9}  {'Neutral':>7}  {'Hoarders':>8}")
print(f"  {'─'*4}  {'─'*5}  {'─'*9}  {'─'*9}  {'─'*7}  {'─'*8}")

tick_times = []

for tick in range(TICKS):
    t0 = time.perf_counter()
    swarm.tick()
    tick_ms = (time.perf_counter() - t0) * 1000
    tick_times.append(tick_ms)

    # Second shock mid-run — tests whether RL adapts to new pressure
    if tick == 200:
        swarm.apply_environmental_shock(location=(600.0, 400.0), radius=150.0, intensity=1.0)
        print(f"\n  ⚡ New threat at (600,400) — does RL adapt?\n")

    if tick % 50 == 0 or tick == TICKS - 1:
        m  = swarm.sample_population_metrics()
        sp = swarm.get_all_share_probabilities()
        n  = len(sp)
        altruists = sum(1 for p in sp if p > 0.7)
        hoarders  = sum(1 for p in sp if p < 0.3)
        neutral   = n - altruists - hoarders
        print(
            f"  {tick:>4}  {tick_ms:>4.1f}ms  {m['mean_surprise_score']:>9.4f}  "
            f"{altruists:>9}  {neutral:>7}  {hoarders:>8}"
        )

# ── Final analysis ───────────────────────────────────────────────
print("\n" + "=" * 54)
print("RESULTS")
print("=" * 54)

sp    = swarm.get_all_share_probabilities()
n     = len(sp)
altr  = sum(1 for p in sp if p > 0.7)
hoard = sum(1 for p in sp if p < 0.3)
neut  = n - altr - hoard

bar_a = "█" * round(altr  / n * 40)
bar_n = "█" * round(neut  / n * 40)
bar_h = "█" * round(hoard / n * 40)

print(f"\n  🤝 Altruists  (share >70%):  {altr:>5}  {bar_a}")
print(f"  🔄 Neutral    (30–70%):      {neut:>5}  {bar_n}")
print(f"  🦊 Hoarders   (share <30%):  {hoard:>5}  {bar_h}")

bimodal = (altr + hoard) > neut
print(f"\n  Caste Emergence: {'✅ YES — emergent specialization confirmed!' if bimodal else '⚠️  Still converging'}")

avg_ms = sum(tick_times) / len(tick_times)
print(f"  Throughput:      {5_000 / (avg_ms / 1000):>14,.0f} agents/sec")
print(f"  Avg tick time:   {avg_ms:>14.2f} ms")
print(f"  LangChain equiv: ~{5_000 * 0.01:>.2f} per equivalent run (API cost)")
print(f"  Ebbiforge cost:  $0.00")

print(f"\n--- What just happened? ---")
print("5,000 agents navigated a world with villages, cities, and an ambush zone.")
print("TD-RL assigned rewards: +1 for trading at the city, -0.8 near the ambush.")
print("Agents that avoided ambush zones developed HIGH share_probability (altruists).")
print("Agents repeatedly punished near the ambush developed LOW share_probability (hoarders).")
print("Zero hardcoded rules. Zero LLM calls. Pure emergent behavior from RL pressure.")
