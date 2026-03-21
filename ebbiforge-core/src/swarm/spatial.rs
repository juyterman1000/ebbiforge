//! Spatial Grid
//!
//! Spatial hashing grid for fast neighbor lookups in the swarm.

use super::SwarmConfig;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Spatial indexing for the swarm
#[pyclass]
pub struct GridMap {
    config: SwarmConfig,
    cell_size: usize,
    // Map Cell(x,y) -> List of Agent IDs
    // O(1) density lookup via cell counts (memory-efficient vs storing full agent lists)
    cell_counts: HashMap<(usize, usize), usize>,
}

#[pymethods]
impl GridMap {
    #[new]
    #[pyo3(signature = (config = None, cell_size = 10))]
    pub fn new(config: Option<SwarmConfig>, cell_size: usize) -> Self {
        GridMap {
            config: config.unwrap_or_default(),
            cell_size,
            cell_counts: HashMap::new(),
        }
    }

    /// Update spatial index from agent positions (clamped to world bounds)
    pub fn update(&mut self, x_coords: Vec<f32>, y_coords: Vec<f32>) {
        self.cell_counts.clear();
        let max_x = self.config.world_width as f32;
        let max_y = self.config.world_height as f32;

        for (x, y) in x_coords.iter().zip(y_coords.iter()) {
            let clamped_x = x.max(0.0).min(max_x);
            let clamped_y = y.max(0.0).min(max_y);
            let cx = (clamped_x as usize) / self.cell_size;
            let cy = (clamped_y as usize) / self.cell_size;
            *self.cell_counts.entry((cx, cy)).or_insert(0) += 1;
        }
    }

    /// Get density at coordinates
    pub fn get_density(&self, x: f32, y: f32) -> usize {
        let cx = (x as usize) / self.cell_size;
        let cy = (y as usize) / self.cell_size;
        *self.cell_counts.get(&(cx, cy)).unwrap_or(&0)
    }
}
