# Ebbiforge Core

**High-performance multi-agent runtime built in Rust with Python bindings.**

Ebbiforge Core is the engine behind the Ebbiforge framework. It is designed for deploying scalable agent societies using data-oriented architecture and biological memory models.

## Build from Source

```bash
cd ebbiforge-core
cargo build --release
```

## Quick Start

```python
import ebbiforge_core as ors

# Initialize a production swarm with 10,000 full-fidelity agents
graph = ors.AgentGraphPy()
swarm = ors.ProductionTensorSwarm(agent_count=10000)

# Add 1,000,000 dormant agents (Tier 1 bitflags, zero context overhead)
dormant = [ors.DormantAgent(id=i, predicted_state=0, wakeup_conditions=i%2) for i in range(1_000_000)]
swarm.add_dormant_agents(dormant)

# Rapidly filter agents via SIMD (promotes 1M agents in under 12ms)
swarm.set_global_triggers(1)
swarm.tick()

# Process promoted agents through the async multiplexer
promoted_agents = swarm.pop_promotions()
for p in promoted_agents:
    graph.spawn_task(f"Task_{p}", ors.HistoryBuffer(), agent_name="Scout")
```

## Architecture

| Module | Description |
|--------|-------------|
| **TensorSwarm Engine** | Struct-of-Arrays (SoA) layout processed by Rayon SIMD threads. |
| **4-Tier LOD System** | Dormant, Simplified, Full, and Heavy compute tiers for 10M+ agents. |
| **Ebbinghaus Memory** | Cosine similarity tracks prediction error, powering biological memory decay. |
| **Pollination RL** | Temporal Difference (TD) learning creates evolving information broker networks. |
| **Zero-Copy Memory** | `Arc<RwLock>` implementations for O(1) multi-threaded history access. |
| **Safety Shield** | Predictive action analysis blocks dangerous behavior prior to execution. |

## Performance

- 10,000,000+ parallel entities via CPU Level-of-Detail
- Sub-millisecond per-agent tick latency
- Zero thread exhaustion via async Tokio/reqwest multiplexing
