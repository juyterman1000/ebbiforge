# hippocampus-sharp-memory

**Brain-inspired memory for AI agents.** Adaptive retention + Kanerva SDM + locality-sensitive hashing. Sub-microsecond semantic lookup at 46M memories. Only remembers what matters.

```bash
pip install hippocampus-sharp-memory
```

```python
from hippocampus_sharp_memory import create_memory

mem = create_memory()
mem.remember("important fact", salience=80.0)    # survives ~368 ticks
mem.remember("trivial noise", salience=1.0)       # gone after ~5 ticks
mem.advance(100)
print(mem.recall("important", top_k=1))           # still here
```

## Why This Exists

Every AI agent framework stores chat history in a list. That's a to-do app pretending to be a brain.

Real brains don't work that way. They:
- **Prioritize** — low-value information fades, critical knowledge stays sharp
- **Strengthen** memories accessed repeatedly (spaced repetition)
- **Associate** related memories into webs (Kanerva SDM)
- **Amplify** emotional/critical events with higher salience
- **Consolidate** frequently-accessed memories during "sleep" cycles

This library does all of that in **Rust**, exposed to Python via zero-copy PyO3 bindings.

## How Salience Works (Read This First)

Salience controls how long a memory survives. The formula is simple:

```
Retention = e^(-age / salience)
Lifetime  ≈ salience × 4.6 ticks    (time until <1% retention)
```

**You decide what a "tick" means** in your app. Call `mem.tick()` once per second, once per message, once per API call — whatever makes sense. Then pick salience values based on how many ticks you want the memory to survive:

| Salience | Lifetime | Use it for |
|----------|----------|------------|
| `1` | ~5 ticks | Throwaway context ("user said hi") |
| `10` | ~46 ticks | Session-level context (current topic) |
| `50` | ~230 ticks | Important facts (user preferences) |
| `100` | ~460 ticks | Business-critical (complaints, errors) |
| `200` | ~920 ticks | Persistent knowledge (learned patterns) |
| `500` | ~2,300 ticks | Near-permanent (compliance events) |

**Example**: If you tick once per second, `salience=60` means the memory survives ~4.6 minutes. If you tick once per message, it survives ~276 messages.

**Emotional tags multiply salience automatically:**

| Tag | Multiplier | Effect |
|-----|-----------|--------|
| `emotional_tag=0` (neutral) | 1.0x | No change |
| `emotional_tag=1` (positive) | 1.2x | 20% longer |
| `emotional_tag=2` (negative) | 1.5x | 50% longer |
| `emotional_tag=3` (critical) | 3.0x | 3x longer |

So `salience=20, emotional_tag=3` becomes effective salience 60 (survives ~276 ticks instead of ~92).

**Spaced repetition**: Every `recall()` multiplies salience by 1.3x. A memory recalled 5 times has its salience boosted 3.7x — it effectively becomes permanent.

## Quick Start

```python
from hippocampus_sharp_memory import create_memory

mem = create_memory()

# Pick salience based on how long you want memories to last
mem.remember("user prefers dark mode", salience=50.0)                         # ~230 ticks
mem.remember("billing complaint about invoice #4821", salience=100.0)          # ~460 ticks
mem.remember("CRITICAL: database at 95% capacity", salience=100.0, emotional_tag=3)  # 3x → ~1380 ticks
mem.remember("user said hello", salience=1.0)                                  # ~5 ticks (throwaway)

# Advance time — this is how forgetting happens
# The "hello" decays away; the critical alert stays sharp
mem.advance(100)

# Semantic recall — finds related memories, not just keyword matches
results = mem.recall("database storage issue", top_k=3)
for r in results:
    print(f"  [{r.retention*100:.0f}%] {r.content}")

# "user said hello" is already forgotten. The critical alert persists.
```

## The Recall-Before-LLM Pattern

The killer use case. **Save 90% on LLM costs:**

```python
def handle_alert(alert_text: str, mem, llm_client):
    # Step 1: Check if we've explained this before
    cached = mem.recall(alert_text, top_k=1)
    if cached and cached[0].retention > 0.3:
        return cached[0].content  # Free! No LLM call needed

    # Step 2: Only call LLM for genuinely new situations
    explanation = llm_client.explain(alert_text)

    # Step 3: Cache the expensive response (survives ~460 ticks)
    mem.remember(
        f"Explanation: {explanation}",
        salience=100.0,
        source="llm_cache",
    )
    return explanation
```

**Why `retention > 0.3`?** At 30% retention, the memory is fading but still relevant. Lower thresholds catch older matches; higher thresholds only return fresh ones. Tune to your domain — 0.3 is a good starting point.

Recurring alerts get answered from memory. Novel situations still go to the LLM. Adaptive retention naturally phases out stale explanations as they age.

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  Python API                       │
│  create_memory() → HippocampusEngine              │
├──────────────────────────────────────────────────┤
│                 Rust Core (PyO3)                  │
│  ┌─────────┐  ┌─────────┐  ┌──────────────────┐ │
│  │ SimHash  │→│ LSH     │→│ Context Scorer    │ │
│  │ 1024-bit │  │ 16 tables│  │ sim+recency+sal  │ │
│  │ address  │  │ O(1)    │  │ +emotion weighting│ │
│  └─────────┘  └─────────┘  └──────────────────┘ │
│  ┌─────────────┐  ┌──────────────────────────┐   │
│  │ Adaptive  │  │ Kanerva SDM              │   │
│  │ Retention │  │ Consolidated Long-Term   │   │
│  └─────────────┘  └──────────────────────────┘   │
│  ┌──────────────────────────────────────────┐    │
│  │ Deduplication (LSH exact-match)          │    │
│  │ Identical content → salience boost       │    │
│  └──────────────────────────────────────────┘    │
├──────────────────────────────────────────────────┤
│          Optional Disk Persistence                │
│  mmap'd records + quota enforcement + compaction  │
└──────────────────────────────────────────────────┘
```

## API Reference

### Factory Functions

```python
from hippocampus_sharp_memory import create_memory, create_persistent_memory

# In-memory (fast, ephemeral) — default 500K capacity
mem = create_memory()

# High-capacity for long-running processes — 1M capacity
mem = create_persistent_memory()
```

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `capacity` | 500K / 1M | Max episodes before oldest are evicted |
| `consolidation_interval` | 100 | Ticks between sleep-replay cycles |
| `recall_reinforcement` | 1.3 | Salience boost per recall (1.0 = no boost) |

### Core Operations

| Method | Description |
|---|---|
| `mem.remember(content, salience, source="", emotional_tag=0)` | Store a memory. Duplicates auto-merge. |
| `mem.recall(query, top_k=5)` | Semantic recall. Returns `List[RecallResult]`. |
| `mem.tick()` | Advance clock by 1. Triggers decay + consolidation. |
| `mem.advance(n)` | Advance clock by `n` ticks at once. |
| `mem.relate(id_a, id_b)` | Create associative link between memories. |
| `mem.recall_related(id, depth=1)` | Follow relationship web. |
| `mem.recall_between(start, end, top_k=10)` | Temporal range query. |
| `mem.stats()` | Returns `HippocampusStats` snapshot. |
| `mem.consolidate_now()` | Force a sleep-replay consolidation cycle. |

### RecallResult Fields

| Field | Type | Description |
|---|---|---|
| `content` | `str` | The memory text |
| `source` | `str` | Origin tag |
| `salience` | `float` | Current importance (grows with recalls) |
| `retention` | `float` | 0.0–1.0, how well-retained right now |
| `age_ticks` | `float` | Ticks since creation |
| `recall_count` | `int` | Times this memory was recalled |
| `consolidated` | `bool` | Promoted to long-term (Kanerva SDM) storage |

## Performance

Benchmarked on a single core (Intel i7-12700K):

| Scale | `remember()` | `recall()` | Memory |
|---|---|---|---|
| 1K memories | 2 us | 8 us | ~1 MB |
| 10K memories | 2 us | 20 us | ~8 MB |
| 100K memories | 3 us | 50 us | ~80 MB |
| 1M memories | 3 us | 120 us | ~800 MB |
| 46M memories | 4 us | 2 us (LSH) | ~37 GB |

The LSH index provides **O(1) query time** regardless of memory count at scale. At 46M memories, recall is actually *faster* than at 1M because the LSH buckets are more selective.

## Advanced Usage

### Spaced Repetition

```python
mem = create_memory()
mem.remember("important pattern", salience=20.0)   # survives ~92 ticks

# Each recall boosts salience by 1.3x
mem.recall("important pattern", top_k=1)  # salience → 26  (~120 ticks)
mem.recall("important pattern", top_k=1)  # salience → 34  (~156 ticks)
mem.recall("important pattern", top_k=1)  # salience → 44  (~202 ticks)
# After 5 recalls: salience ~74, effectively permanent
```

### Automatic Deduplication

```python
mem.remember("server alert: CPU at 95%", salience=20.0)
mem.remember("server alert: CPU at 95%", salience=20.0)  # same content
mem.remember("server alert: CPU at 95%", salience=20.0)  # again!

assert mem.episode_count == 1  # Only 1 episode stored
# Salience was boosted, not duplicated
```

### Relationship Graphs

```python
mem.remember("billing complaint", salience=50.0)       # id=0
mem.remember("escalation to manager", salience=50.0)    # id=1
mem.remember("legal threat received", salience=100.0)   # id=2

mem.relate(0, 1)  # complaint → escalation
mem.relate(1, 2)  # escalation → legal

# Follow the chain
related = mem.recall_related(0, depth=2)
# Returns: [escalation, legal threat]
```

### Consolidation (Sleep Replay)

Memories that are both high-retention AND frequently recalled (2+ times) get consolidated into the permanent Kanerva SDM store. This happens automatically every `consolidation_interval` ticks, or on demand:

```python
# Automatic: happens during tick() / advance()
mem.advance(200)

# Manual: force it now
report = mem.consolidate_now()
print(report)  # "Consolidation: evicted=12, consolidated=3, surviving=85"
```

Consolidated memories are harder to lose — they're stored redundantly in both the hippocampal buffer and the Kanerva SDM.

## Part of the Ebbiforge Ecosystem

`hippocampus-sharp-memory` is the standalone memory engine extracted from [Ebbiforge](https://github.com/juyterman1000/ebbiforge) — a full AI agent framework with:

- **100M-agent swarm simulation** (Rust tensor engine)
- **Compliance & PII redaction** (OWASP, rate limiting, audit trails)
- **Self-evolution** (Darwinian agent selection, metacognition)
- **Latent world model** (predictive planning, diffusion predictor)

If you need just memory → `pip install hippocampus-sharp-memory`
If you need the full stack → `pip install ebbiforge`

## License

MIT
