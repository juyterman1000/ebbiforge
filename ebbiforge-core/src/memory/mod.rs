//! HippocampusEngine — Brain-Inspired Memory for AI Agents
//!
//! Combines four scientific foundations into a novel memory system:
//!
//! 1. **Complementary Learning Systems** (McClelland et al., 1995)
//!    - Hippocampus: fast episodic storage (ring buffer)
//!    - Neocortex:   slow semantic patterns (Kanerva SDM)
//!
//! 2. **Ebbinghaus Forgetting Curve** (1885)
//!    - Retention = e^(-t / Salience)
//!    - High-salience memories survive; routine ones decay
//!
//! 3. **Kanerva Sparse Distributed Memory** (1988)
//!    - O(1) associative recall via XOR + POPCNT
//!    - Sub-linear similarity search over millions of entries
//!
//! 4. **Spaced Recall Reinforcement** (ICLR 2025)
//!    - Every recall() multiplies salience by a reinforcement factor
//!    - Memories that matter get stronger automatically

pub mod episode;
pub mod kanerva;
pub mod hippocampus;
pub mod consolidation;
pub mod disk;
pub mod lsh;

pub use episode::Episode;
pub use kanerva::KanervaSDM;
pub use hippocampus::HippocampusEngine;
