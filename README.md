<div align="center">
  <img src="logo.svg" alt="Ebbiforge" width="280">

  <h3>Ebbiforge</h3>

  <p><strong>The high-performance multi-agent framework.</strong><br>
  100M agents in Rust. Selective LLM reasoning. Enterprise compliance built-in.</p>

  [![CI](https://github.com/juyterman1000/ebbiforge/actions/workflows/ci.yml/badge.svg)](https://github.com/juyterman1000/ebbiforge/actions)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org/)
  [![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
</div>

<p align="center">
  <img src="demo/hero_banner.png" alt="Ebbiforge — Multi-agent swarm reacting to live data" width="800">
</p>

---

## Why Ebbiforge?

| | LangChain | CrewAI | **Ebbiforge** |
|---|:---------:|:------:|:-----------------:|
| **Agent scale** | ~50 | ~10 | **100,000,000** |
| **Cost per step** | $0.01+ | $0.01+ | **$0.00** (Rust) |
| **Compliance** | ❌ | ❌ | **PII + GDPR + Audit** |
| **Belief provenance** | ❌ | ❌ | **Full chain** |
| **World model** | ❌ | ❌ | **Learned dynamics** |
| **Evolution** | ❌ | ❌ | **Darwinian genetics** |

**The key idea**: 99.99% of agents run in Rust at zero cost. Only when the swarm detects something interesting does an LLM get called. This is *selective reasoning* — the LLM is a consultant, not a worker.

---

## Installation

Pick **one** of the two options below:

### Option A: pip install (recommended)

```bash
pip install ebbiforge
```

Everything is included — the Rust engine comes pre-compiled in the wheel. No Rust toolchain needed.

### Option B: Build from source (for contributors)

```bash
git clone https://github.com/juyterman1000/ebbiforge.git
cd ebbiforge && pip install maturin && maturin develop --release
```

Requires Python 3.9+ and Rust 1.75+. Use this if you want to modify the Rust engine or contribute to the project.

---

## Quick Start

### Hello Swarm (5 lines)

```python
from ebbiforge import Agent, Swarm, Task

swarm = Swarm()
swarm.add(Agent(name="Researcher"))
swarm.add(Agent(name="Analyst"))
result = swarm.run(Task("Analyze emerging AI trends", pipeline=True))

# Every claim tracked with source attribution
provenance = swarm.get_belief_provenance(result)
if provenance.has_unverified:
    print("⚠️ Pipeline contains unverified claims!")
```

### 50K Agents with Selective Reasoning

```python
import ebbiforge_core as ebbi

swarm = ebbi.TensorSwarm(agent_count=50_000,
    world_config=ebbi.WorldModelConfig(ebbinghaus_decay_rate=0.1, grid_size=(200, 200)))

for tick in range(1000):
    swarm.tick()
    scores = swarm.surprise_scores  # Per-agent surprise (0.0–1.0)
    surprised = sum(1 for s in scores if s > 0.3)
    if surprised / 50_000 > 0.05:  # 5% threshold
        print(f"🧠 Tick {tick}: Anomaly! Call LLM for {surprised} agents")
    # The other 995+ ticks? FREE. No LLM. Pure Rust.
```

### Connect Any Data Source

```python
from ebbiforge.connectors import HTTPPoller, Signal

def parse(data: dict) -> list[Signal]:
    return [Signal(source="my-api", value=data["metric"] / 100.0)]

feed = HTTPPoller(url="https://my-api.com/data", transform=parse)
```

### Enterprise Compliance

```python
import ebbiforge_core as ebbi

compliance = ebbi.ComplianceEngine()
result = compliance.check_action("agent-1", "send_email", "SSN: 123-45-6789")
# → BLOCKED: PII detected
```

---

## World Model Training

Ebbiforge includes a **learned dynamics model** that predicts agent state transitions. The model learns causal structure from logged trajectories — no pre-trained weights needed.

### Collect Trajectories & Train

```python
import ebbiforge_core as ebbi

# 1. Create a global trajectory buffer (all agents log into ONE dataset)
buf = ebbi.TrajectoryBuffer(capacity=100_000)

# 2. Run the swarm, sample agent transitions each tick
swarm = ebbi.TensorSwarm(agent_count=1_000_000)
for tick in range(1000):
    swarm.tick()
    # Record (state_t, action_t, state_{t+1}) for sampled agents
    # buf.record(state_vec, action_vec, next_state_vec)

# 3. Train the dynamics model (AdamW, cosine LR, validation split)
predictor = ebbi.AutoregressivePredictor()
stats = predictor.train(
    buf.to_json(),
    epochs=100,
    learning_rate=0.01,
    batch_size=256,
    val_split=0.1
)

# 4. Persist learned weights
predictor.save_weights("world_model.safetensors")
print(f"Final loss: {stats[-1].train_loss:.6f}")
```

### Architecture

```
state_t + action_t → [Linear + GELU] → [Residual + GELU] → [Linear] → state_{t+1}
                      ↑ AdamW optimizer, cosine LR decay, early stopping
```

The model learns the **Markovian transition function**: `state_{t+1} = f(state_t, action_t)`. Neighbor interactions (surprise propagation, flocking, pheromone steering) are already encoded in each agent's state by the Rust engine.

---

## CLI

```bash
ebbiforge demo             # Interactive 10K agent demo
ebbiforge benchmark        # Performance scaling 1K → 100K
ebbiforge example hello    # Run a built-in example
ebbiforge version          # System info
```

---

## Architecture: Two-Tier Intelligence

```
┌────────────────────────────────────────────────┐
│          Tier 2: Conscious Brain               │
│  Python + LLM (Gemini)    EXPENSIVE, SMART     │
│  ReAct loop + tool use    Called only on anomaly│
├────────────────────────────────────────────────┤
│          Tier 1: Nervous System                │
│  Pure Rust (zero-cost)    FAST, FREE           │
│  Spatial hash + SIRS      100K-100M agents     │
│  Darwinian evolution      ~0.01ms per agent    │
│  Learned world model      AdamW dynamics MLP   │
└────────────────────────────────────────────────┘
```

Learn more: **[Architecture Guide](docs/architecture.md)** | **[Quick Start](docs/quickstart.md)**

---

## What's Inside

### Core Engine (`ebbiforge-core/`)
- **Agent Orchestration**: ReAct loop, handoffs, sequential/parallel/loop workflows
- **Memory**: In-process (SharedMemory), Redis/Dragonfly (KV), Qdrant (vector)
- **Swarm**: 100M agents via mmap, Fibonacci spatial hash, SIRS epidemiology, pheromones
- **World Model**: Latent encoder (fastembed ONNX), autoregressive dynamics (Candle MLP), trajectory buffer
- **Safety**: PredictiveSafetyShield, PII redaction, GDPR, rate limiting, policy engine
- **Intelligence**: MetaCognition, IntrospectionEngine, CuriosityModule

### Python Framework (`ebbiforge/`)
- **Primitives**: Agent, Swarm, Runtime, Task, Memory
- **Connectors**: DataSource, HTTPPoller, WebhookReceiver, RSSFeed
- **Outputs**: OutputSink, ConsoleOutput, WebhookOutput, JSONFileOutput
- **CLI**: demo, benchmark, example, version

### Examples (`examples/`)
| # | Name | What It Shows |
|---|------|---------------|
| 1 | [Hello Swarm](examples/01_hello_swarm.py) | Multi-agent pipeline + belief provenance |
| 2 | [Evolution](examples/02_evolution.py) | 10K agents with Darwinian natural selection |
| 3 | [Live Data](examples/03_live_data.py) | Generic HTTPPoller → swarm signals |
| 4 | [Selective Reasoning](examples/04_selective_reasoning.py) | Two-tier: 50K Rust agents + LLM on anomaly |
| 5 | [Compliance](examples/05_compliance.py) | PII detection + GDPR + audit trail |

---

## Performance

Verified on standard hardware (details in [BENCHMARKS.md](BENCHMARKS.md)):

| Agents | Avg Tick | Throughput |
|--------|----------|------------|
| 10,000 | ~0.5ms | 20M agents/sec |
| 100,000 | ~5ms | 20M agents/sec |
| 10,000,000 | ~500ms | 20M agents/sec |

---

## Contributing

We welcome contributions! Especially around:
- New `DataSource` connectors for different domains
- New `OutputSink` adapters (Slack, Discord, PagerDuty)
- SIMD optimizations for the Tier 1 engine
- Example scripts for specific use cases

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## License

[MIT License](LICENSE)
