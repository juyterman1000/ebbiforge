#!/usr/bin/env python3
"""
Hippocampus Memory — Quick Start
=================================

Demonstrates the core memory lifecycle:
  1. Storing memories with varying salience
  2. Semantic recall (not keyword matching)
  3. Ebbinghaus forgetting over time
  4. Spaced repetition (recall strengthens memory)
  5. Automatic deduplication
  6. Relationship graphs
"""

from hippocampus_sharp_memory import create_memory

# ── Create a memory engine ─────────────────────────────────────────────
mem = create_memory(capacity=10_000, recall_reinforcement=1.3)
print("🧠 Hippocampus Memory — Quick Start\n")

# ── 1. Store memories with different importance levels ─────────────────
print("1️⃣  Storing memories...")
mem.remember("user prefers dark mode and large fonts", salience=20.0)
mem.remember("billing complaint about invoice #4821", salience=60.0)
mem.remember("server CPU spike to 95% at 2am", salience=40.0, emotional_tag=2)
mem.remember("CRITICAL: database approaching disk limit", salience=90.0, emotional_tag=3)
mem.remember("weather is nice today", salience=1.0)  # Low importance — will decay
print(f"   Stored {mem.episode_count} memories\n")

# ── 2. Semantic recall ─────────────────────────────────────────────────
print("2️⃣  Semantic recall: 'storage disk problem'")
results = mem.recall("storage disk problem", top_k=3)
for i, r in enumerate(results):
    print(f"   #{i+1} [{r.retention*100:.0f}% retained] {r.content[:60]}")
print()

# ── 3. Ebbinghaus forgetting ──────────────────────────────────────────
print("3️⃣  Advancing time to trigger forgetting...")
before = mem.episode_count
for _ in range(200):
    mem.tick()  # Each tick applies Ebbinghaus decay
after = mem.episode_count
print(f"   Before: {before} memories → After: {after} memories")
print(f"   Low-salience memories decayed naturally\n")

# ── 4. Spaced repetition ──────────────────────────────────────────────
print("4️⃣  Spaced repetition (recall strengthens memory)")
r1 = mem.recall("billing complaint", top_k=1)
s1 = r1[0].salience if r1 else 0
r2 = mem.recall("billing complaint", top_k=1)
s2 = r2[0].salience if r2 else 0
r3 = mem.recall("billing complaint", top_k=1)
s3 = r3[0].salience if r3 else 0
print(f"   Salience after recalls: {s1:.1f} → {s2:.1f} → {s3:.1f}")
print(f"   Each recall made the memory harder to forget\n")

# ── 5. Deduplication ──────────────────────────────────────────────────
print("5️⃣  Automatic deduplication")
count_before = mem.episode_count
mem.remember("CRITICAL: database approaching disk limit", salience=90.0, emotional_tag=3)
mem.remember("CRITICAL: database approaching disk limit", salience=90.0, emotional_tag=3)
count_after = mem.episode_count
print(f"   Stored same message 2 more times: {count_before} → {count_after} episodes")
print(f"   Duplicates merged, salience boosted\n")

# ── 6. Stats ──────────────────────────────────────────────────────────
stats = mem.stats()
print("📊 Engine Stats:")
print(f"   Episodes:     {stats.episode_count}")
print(f"   Consolidated: {stats.consolidated_count}")
print(f"   Avg salience: {stats.avg_salience:.1f}")
print(f"   Total recalls: {stats.total_recalls}")
print(f"   Total evicted: {stats.total_evicted}")

print("\n✅ Done! Only remembers what matters.")
