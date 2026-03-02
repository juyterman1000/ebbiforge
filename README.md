<div align="center">
  <img src="logo.svg" alt="OpenRustSwarm" width="280">

  <h3>OpenRustSwarm</h3>

  <p>A high-performance research substrate for large-scale agent simulations.</p>

  [![CI](https://github.com/juyterman1000/openrustswarm/actions/workflows/ci.yml/badge.svg)](https://github.com/juyterman1000/openrustswarm/actions)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org/)
  [![Next.js](https://img.shields.io/badge/next.js-15-black.svg)](https://nextjs.org/)
</div>

<p align="center">
  <img src="demo/hero_banner.png" alt="OpenRustSwarm — BTC, ETH, SOL organism clusters reacting to live market data" width="800">
</p>

---

## The Problem: Scaling Complexity in Agent Simulations

Most agent-based simulations struggle with the $O(N^2)$ neighbor lookup problem and high memory overhead per agent. When scaling to millions of entities, traditional object-oriented patterns or even standard Entity-Component-System (ECS) approaches often hit wall-clock or memory limits on consumer hardware.

OpenRustSwarm is a research project exploring how to use **Level of Detail (LOD)** strategies—common in 3D rendering but less so in agent logic—to simulate up to 10,000,000 agents on a single workstation.

## Technical Strategy: 4-Tier LOD Architecture

We categorize agents by their "Criticality" and "Surprise Score" to determine how much compute resource they consume.

```mermaid
graph TD
    T1[Tier 1: Dormant - 9.0M Agents] -->|State Trigger| T2
    T2[Tier 2: Simplified - 0.8M Agents] -->|Anomaly Detection| T3
    T3[Tier 3: Full Tensor - 0.2M Agents] -->|Critical Threshold| T4
    T4[Tier 4: Heavy - 10-100 Agents]

    subgraph "Nervous System (Rust Engine)"
    T1 -.->|Packed Bitfields / mmap| T1
    T2 -.->|SIMD-Optimized Physics| T2
    T3 -.->|Tensor-Based Decision Logic| T3
    end

    subgraph "Conscious Action (Optional Integration)"
    T4 -.->|LLM-Narrated / OpenClaw| T4
    end
```

### Key Optimizations

- **mmap-Backed Dormant Pool**: T1 agents are stored in a memory-mapped array with a 256-bit footprint per agent, minimizing the resident set size (RSS).
- **Spatial Hash Grid**: We use a zero-copy spatial hash for $O(1)$ neighbor queries, avoiding expensive bridge-crossing between WASM and JavaScript in browser environments.
- **SIRS Epidemiology**: Instead of basic "health," we use a Susceptible-Infected-Recovered system where "Surprise" from data volatility acts as the infectious agent.
- **Darwinian Genetics**: A custom genetic crossover engine allows for emergent behavioral shifts over thousands of generations.

---

## Performance & Benchmarks

We have verified a stable 10,000,000 agent simulation (1M active) on standard hardware with a throughput of **~20.7 Million updates per second**.

Detailed methodology, test environment specs, and instructions to reproduce these numbers can be found in [BENCHMARKS.md](BENCHMARKS.md).

---

## Use Cases

1.  **Collective Intelligence Research**: Testing how high-frequency data shocks propagate through massive populations.
2.  **Simulation Engineering**: A reference implementation for scaling PyO3/Rust simulations with `mmap`.
3.  **Visualization Tech**: Stress-testing WebGL and Instanced Rendering in Next.js/WASM environments.

---

## Quick Start

```bash
git clone https://github.com/juyterman1000/openrustswarm.git
cd openrustswarm/web
npm install
npm run dev
```

*Note: The browser demo is limited to 200,000 agents to maintain 60fps on typical mobile/web hardware.*

---

## Roadmap

- [x] **Rust Core**: Memory-mapped LOD system, spatial hash, and SIRS logic.
- [x] **Evolution**: Genetic crossover and point mutation engine.
- [x] **WASM Bridge**: High-frequency data injection from CoinGecko/GitHub.
- [ ] **Methodology Paper**: A detailed write-up of the LOD strategy for agent simulations.
- [ ] **WebAudio**: R0-driven harmonic chord synthesis.
- [ ] **Cross-Instance Sync**: Pheromone field diffusion over WebSockets.

---

## Contributing

We welcome technical contributions, especially around SIMD optimizations for the T2/T3 tiers and new data feed integrations. 

See [CONTRIBUTING.md](CONTRIBUTING.md) for architectural deep dives.

---

## License

[MIT License](LICENSE)
