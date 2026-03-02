# Contributing to OpenRustSwarm

We are looking for engineers interested in pushing the boundaries of large-scale agent simulations. Here is how to get involved.

## Development Setup

### Rust Engine
```bash
git clone https://github.com/juyterman1000/openrustswarm.git
cd openrustswarm/openrustswarm-core
cargo build --release
cargo test
```

### Web Dashboard
```bash
cd web
npm install
npm run dev
```

## Architectural Deep Dives

### 1. The LOD Tiering System (`lod.rs`)
The core scaling strategy relies on four tiers of agent fidelity.
- **T1 (Dormant)**: Stored in bit-packed fields within a memory-mapped array. Update cost is effectively zero until a dormancy trigger is hit.
- **T2 (Simplified)**: Uses a centroid-based physics approximation.
- **T3 (Full Tensor)**: Full neighbor-to-neighbor interaction using the spatial hash grid.
- **T4 (Heavy)**: Reserved for agents requiring high-fidelity state histories or external LLM introspection.

**Contribution Area**: We need better heuristics for "Surprise-driven" transitions between T2 and T3 to avoid oscillation.

### 2. Spatial Hash Grid (`spatial.rs`)
To avoid $O(N^2)$ complexity, we use a uniform spatial grid. 
- **Zero-Copy**: The grid is designed for minimal allocation during the tick loop.
- **WASM Bridge**: The grid state is shared with the WebGL renderer via a shared memory buffer.

**Contribution Area**: Implementing multi-layered grids for varying agent radii.

## Good First Issues

| Task | Area | Difficulty |
|------|---------|------------|
| Add a new data feed integration | Rust/Python | 🟢 Easy |
| Optimize SIMD kernels for T2 centroid logic | Rust | 🟡 Medium |
| Implement WebAudio harmonic synthesis | TS/WASM | 🟡 Medium |
| Refactor `lod.rs` bitflag packing for efficiency | Rust | 🔴 Hard |

## Technical Standards

- **Performance First**: Any PR that regresses throughput (as measured by `test_10m_scale.py`) will require detailed justification.
- **No Unsafe**: Avoid `unsafe` unless necessary for `mmap` or SIMD intrinsics. Document all such uses.

## Pull Request Process

1. Fork the repository.
2. Create a feature branch (`git checkout -b feat/your-fix`).
3. Benchmark your changes against the `main` branch.
4. Open a PR with a description of the technical trade-offs made.
