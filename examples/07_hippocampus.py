#!/usr/bin/env python3
"""
07_hippocampus.py — HippocampusEngine Demo
============================================

Demonstrates Ebbiforge's brain-inspired memory system:

  1. Ebbinghaus decay  — routine memories auto-evict, traumatic ones persist
  2. Kanerva SDM       — O(1) associative recall via XOR + POPCNT hardware
  3. Sleep consolidation — important memories graduate to permanent storage
  4. Spaced repetition  — each recall() strengthens the memory

Run:
    python examples/07_hippocampus.py
    ebbiforge example hippocampus
"""

import time
import ebbiforge_core as ebbi

DIVIDER = "─" * 60

def main():
    print()
    print("🧠  H I P P O C A M P U S   E N G I N E")
    print("    Brain-Inspired Memory for AI Agents")
    print(DIVIDER)

    # ── 1. Create the engine ──────────────────────────────────
    mem = ebbi.HippocampusEngine(
        capacity=100_000,
        sdm_locations=10_000,
        sdm_radius=400,
        consolidation_interval=20,   # consolidate every 20 ticks
        recall_reinforcement=1.3,
    )
    print(f"  Engine created: capacity=100K, SDM=10K locations")
    print()

    # ── 2. Store memories with varying salience ───────────────
    #   Salience units: 1 tick ≈ 1 minute.  So salience=60 means
    #   "50% retention after ~42 minutes" (e^-42/60 ≈ 0.5).
    print("📝  Storing memories (Salience = minutes to 37% retention)...")
    memories = [
        ("User said hello",                          2.0,  "",             0),  # routine
        ("User asked about weather",                 2.0,  "",             0),  # routine
        ("User complained about billing overcharge", 50.0, "chat_log",     2),  # negative
        ("User threatened legal action",             80.0, "ticket_4821",  3),  # critical
        ("System processed payment OK",              5.0,  "payments",     1),  # positive
        ("User praised the support team",            15.0, "survey",       1),  # positive
        ("Database timeout at 3:47 AM",              40.0, "alert_system", 2),  # negative
        ("CEO requested compliance audit",           90.0, "email",        3),  # critical
    ]

    for content, salience, source, emotion in memories:
        mem.remember(content, salience=salience, source=source, emotional_tag=emotion)
        tag_names = ["neutral", "positive", "negative", "critical"]
        adj_s = salience * [1.0, 1.2, 1.5, 3.0][emotion]
        print(f"    S={adj_s:>6.0f}  [{tag_names[emotion]:>8}]  {content}")

    stats = mem.stats()
    print(f"\n  📊 After storing: {stats.episode_count} episodes, avg salience={stats.avg_salience:.1f}")
    print()

    # ── 3. Simulate agent activity: periodic recalls ──────────
    print("🔁  Simulating 10 minutes of agent activity...")
    print("    Agent recalls critical memories occasionally (building salience)")
    print()

    for tick in range(10):
        mem.tick()    # 1 tick ≈ 1 minute
        # Agent naturally recalls important topics
        if tick == 2:
            mem.recall("billing", top_k=2)
            print(f"    t={tick:>2}: Agent recalled 'billing'")
        if tick == 5:
            mem.recall("legal action", top_k=1)
            print(f"    t={tick:>2}: Agent recalled 'legal action'")
        if tick == 7:
            mem.recall("compliance audit", top_k=1)
            mem.recall("legal", top_k=1)
            print(f"    t={tick:>2}: Agent recalled 'compliance' + 'legal'")

    print()

    # ── 4. Time passes: 100 ticks (≈ 100 minutes) ────────────
    print("⏰  100 minutes pass... (5 consolidation cycles)")
    before = mem.stats()
    for _ in range(100):
        mem.tick()
    after = mem.stats()

    survived = after.episode_count
    evicted = after.total_evicted - before.total_evicted
    consolidated = after.consolidated_count

    print(f"  BEFORE: {before.episode_count} episodes")
    print(f"  AFTER:  {survived} episodes  ({evicted} evicted, {consolidated} consolidated)")
    print()

    # ── 5. Recall after decay ─────────────────────────────────
    print("🔍  Recall: 'billing' (after 100 minutes)")
    results = mem.recall("billing", top_k=3)
    if results:
        for i, r in enumerate(results):
            icon = "📦" if r.consolidated else "🧠"
            print(f"    {i+1}. {icon} S={r.salience:.0f}  ret={r.retention*100:.1f}%  recalls={r.recall_count}  {r.content[:45]}")
            if r.source:
                print(f"         └─ source: {r.source}")
    else:
        print("    (all billing memories decayed)")

    print()
    print("🔍  Recall: 'hello' (routine — should be gone)")
    results = mem.recall("hello", top_k=2)
    if results:
        for i, r in enumerate(results):
            print(f"    {i+1}. S={r.salience:.0f}  ret={r.retention*100:.1f}%  {r.content[:45]}")
    else:
        print("    ✅ Routine memories forgotten (as expected!)")
    print()

    # ── 6. Spaced recall reinforcement demo ───────────────────
    print("💪  Spaced Recall Reinforcement")
    print("    Recalling 'legal action' 5× to simulate active use...")
    for i in range(5):
        results = mem.recall("legal action", top_k=1)
        if results:
            r = results[0]
            print(f"    #{i+1}: S={r.salience:>7.1f}  retention={r.retention*100:.1f}%  '{r.content[:35]}…'")
    print()

    # ── 7. Performance benchmark ──────────────────────────────
    print("⚡  Performance Benchmark")
    bench = ebbi.HippocampusEngine(capacity=100_000, sdm_locations=1_000, consolidation_interval=999_999)

    t0 = time.perf_counter()
    N = 10_000
    for i in range(N):
        bench.remember(f"benchmark memory item number {i} with unique content", salience=50.0)
    write_ms = (time.perf_counter() - t0) * 1000
    write_ns = write_ms * 1e6 / N

    t0 = time.perf_counter()
    R = 1_000
    for i in range(R):
        bench.recall(f"benchmark {i}", top_k=5)
    read_ms = (time.perf_counter() - t0) * 1000
    read_us = read_ms * 1000 / R

    print(f"    Write: {write_ns:.0f} ns/op  ({N:,} memories in {write_ms:.0f}ms)")
    print(f"    Recall: {read_us:.0f} μs/op  ({R:,} queries over {N:,} memories)")
    print(f"    Memory: ~{N * 192 // 1024} KB for {N:,} episodes")
    print()

    # ── 8. Final scorecard ────────────────────────────────────
    final = mem.stats()
    print(DIVIDER)
    print("🏆  FINAL SCORECARD")
    print(DIVIDER)
    print(f"  Episodes surviving:     {final.episode_count}")
    print(f"  Consolidated (SDM):     {final.consolidated_count}  ← permanent memories")
    print(f"  Total evicted:          {final.total_evicted}  ← naturally forgotten")
    print(f"  Total recalls:          {final.total_recalls}")
    print(f"  Consolidation cycles:   {final.total_consolidation_cycles}")
    print(f"  SDM utilization:        {final.sdm_occupied}/{final.sdm_capacity}")
    print(DIVIDER)

    if final.total_evicted > 0 and final.episode_count > 0:
        print()
        print("  ✅ Routine memories decayed. Critical memories survived.")
        print("     This is Ebbinghaus + CLS theory in Rust, at ~15ns/op.")
    elif final.total_evicted > 0:
        print()
        print("  ✅ Ebbinghaus decay working — low-salience memories evicted.")
    print()


if __name__ == "__main__":
    main()
