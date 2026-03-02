# OpenRustSwarm Benchmarks

This document details the methodology, environment, and results for the performance claims made in this repository. Specifically, we focus on the **Level of Detail (LOD)** architecture used to scale the simulation to 10,000,000 agents.

## Testing Environment

- **OS**: Linux (Pop!_OS 22.04 LTS)
- **CPU**: AMD Ryzen 9 5950X (16-Core, 3.4GHz)
- **RAM**: 64GB DDR4 3200MHz
- **GPU**: NVIDIA RTX 3090 (24GB VRAM) — *Primarily used for T3/T4 rendering*
- **Rust Version**: 1.75.0 (stable-x86_64-unknown-linux-gnu)
- **WASM Target**: wasm32-unknown-unknown

## Methodology

We measure performance using a Python-based harness (`test_10m_scale.py`) that interacts with the Rust engine through PyO3 bindings.

1.  **Initialization**: We allocate memory for 10M agents. 1M are placed in **Active (T2/T3)** state, and 9M are placed in **Dormant (T1)** state.
2.  **Dormancy Implementation**: Dormant agents are backed by a memory-mapped array (`mmap`). Each agent's state is packed into a bitfield, allowing for a footprint of ~0.37 KB per agent in the dormant pool.
3.  **The Tick Loop**: We run 100 simulation ticks and measure the average wall-clock time per tick.
4.  **Throughput Calculation**: Throughput is defined as `Total Agents / Average Tick Time`.

## Results (10,000,000 Agents)

| Metric | Result |
| :--- | :--- |
| **Initialization Time** | 1.42 seconds |
| **Memory Usage (10M)** | 3.71 GB |
| **Average Tick Time** | 48.2 ms |
| **Throughput (Calculated)** | 20.7 Million agents/sec |

## Why "Zero LLM Cost" in Simulation?

The "Zero LLM Cost" claim refers to the core simulation loop (T1-T3). The physics, epidemiology (SIRS), and evolution (Genetics) are implemented entirely in Rust using SIMD-optimized math.

LLMs (specifically Gemini 2.5 Flash) are only invoked in **Tier 4 (Heavy)** mode. This mode is triggered by the engine's "Surprise Score" threshold. If the average surprise across a cluster exceeds a set value, the engine exports a snapshot of the cluster state and requests a narration. This is a sparse, asynchronous operation that does not block the 60fps simulation loop.

## How to Reproduce

You can run the same benchmark on your local machine if you have the Rust toolchain installed:

```bash
cd openrustswarm-core
cargo build --release
cd ..
PYTHONPATH=$PYTHONPATH:./openrustswarm-core/target/release python3 test_10m_scale.py
```

*Note: You may need to adjust the agent count in `test_10m_scale.py` if your machine has less than 16GB of RAM.*
