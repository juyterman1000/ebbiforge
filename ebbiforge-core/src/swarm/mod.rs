//! Swarm Engine
//!
//! Handles massive-scale agent simulation using tensor-based state (SoA).
//! Inspired by modern simulation frameworks.

pub mod promoter;
pub mod spatial;
pub mod pollination;
pub mod tensor_engine;
pub mod lod;
pub mod swarm_engine;
pub mod grid;
pub mod master_pipeline;
pub mod pheromone;
pub mod py_api;
pub mod mmap_pool;
pub mod watcher;
pub mod scale_test;
pub mod criticality_test;
pub mod evolution_test;

use pyo3::prelude::*;

pub use promoter::PromotionLogic;
pub use spatial::GridMap;
pub use tensor_engine::TensorSwarm;
pub use lod::{DormantAgent, SimplifiedPool, ProductionTensorSwarm};
pub use master_pipeline::{PropagationConfig, SwarmEvolutionConfig};

/// Configuration for the Swarm Engine
#[derive(Clone, Debug)]
#[pyclass]
pub struct SwarmConfig {
    #[pyo3(get, set)]
    pub population_size: usize,
    #[pyo3(get, set)]
    pub world_width: usize,
    #[pyo3(get, set)]
    pub world_height: usize,
    #[pyo3(get, set)]
    pub max_threads: usize,
}

#[pymethods]
impl SwarmConfig {
    #[new]
    #[pyo3(signature = (population_size = 100000, world_width = 1000, world_height = 1000, max_threads = 8))]
    pub fn new(
        population_size: usize,
        world_width: usize,
        world_height: usize,
        max_threads: usize,
    ) -> Self {
        SwarmConfig {
            population_size,
            world_width,
            world_height,
            max_threads,
        }
    }
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self::new(100_000, 1000, 1000, 8)
    }
}
