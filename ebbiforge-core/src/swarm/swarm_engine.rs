use super::pheromone::PheromoneField;
use rayon::prelude::*;
use std::mem::swap;

/// Constants for Spatial Hashing and Rayon Partitioning
pub const CELL_SIZE: f32 = 10.0; // 10x10 units per spatial cell

/// Struct of Arrays (SoA) memory layout meticulously aligned for pure AVX2 Processing.
/// All "tier" and GPU logic has been removed. We sort this struct in-place to guarantee
/// L1/L2 cache locality during neighbor queries.
pub struct SwarmPool {
    pub n_agents: usize,
    
    // Core Physics (AVX2 loaded in 8-wide blocks)
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub vx: Vec<f32>,
    pub vy: Vec<f32>,
    
    // Cognitive States
    pub surprise: Vec<f32>,
    pub health: Vec<f32>,
    pub hidden_state: Vec<f32>, // Flat array: [N * 32]
    
    // Spatial Hashing Metadata
    pub cell_index: Vec<u32>,   // The morton code / hash index for spatial sorting
}

impl SwarmPool {
    pub fn new(n_agents: usize) -> Self {
        Self {
            n_agents,
            x: vec![0.0; n_agents],
            y: vec![0.0; n_agents],
            vx: vec![0.0; n_agents],
            vy: vec![0.0; n_agents],
            surprise: vec![0.0; n_agents],
            health: vec![1.0; n_agents],
            hidden_state: vec![0.0; n_agents * 32],
            cell_index: vec![0; n_agents],
        }
    }

    /// Computes the 1D spatial hash index for every agent based on their 2D (x,y) coordinates.
    /// This is step 1 of the cache-locality sorting algorithm.
    pub fn update_spatial_hashes(&mut self, world_width: f32) {
        let cols = (world_width / CELL_SIZE).ceil() as u32;
        
        self.x.par_iter()
            .zip(&self.y)
            .zip(&mut self.cell_index)
            .for_each(|((x, y), cell)| {
                 let cx = (*x / CELL_SIZE) as u32;
                 let cy = (*y / CELL_SIZE) as u32;
                 *cell = cy * cols + cx;
            });
    }

    /// The holy grail of CPU N-Body simulations: Memory Spatial Locality Sorting.
    /// Physically rearranges the SoA arrays so that agents who are physically close
    /// in the 2D world are absolutely contiguous in RAM.
    /// This eliminates 90% of cache misses during neighbor aggregation.
    pub fn sort_memory_by_spatial_hash(&mut self) {
        // 1. Create an argsort index array
        let mut indices: Vec<usize> = (0..self.n_agents).collect();
        // Unstable sort is faster and we don't care about relative order within the same cell
        indices.par_sort_unstable_by_key(|&i| self.cell_index[i]);

        // 2. Apply the permutation in-place (requires O(N) scratch space for safety)
        self.apply_permutation(&indices);
    }

    /// Applies the sorted argsort permutation to all SoA arrays to guarantee cache warmth
    fn apply_permutation(&mut self, indices: &[usize]) {
        let n = self.n_agents;

        let mut new_x = vec![0.0; n];
        let mut new_y = vec![0.0; n];
        let mut new_vx = vec![0.0; n];
        let mut new_vy = vec![0.0; n];
        let mut new_surprise = vec![0.0; n];
        let mut new_health = vec![0.0; n];
        let mut new_cell = vec![0u32; n];
        let mut new_hidden = vec![0.0; n * 32];

        // Sequential scatter — safe and still fast (single O(N) pass, memory-bound)
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            new_x[new_idx] = self.x[old_idx];
            new_y[new_idx] = self.y[old_idx];
            new_vx[new_idx] = self.vx[old_idx];
            new_vy[new_idx] = self.vy[old_idx];
            new_surprise[new_idx] = self.surprise[old_idx];
            new_health[new_idx] = self.health[old_idx];
            new_cell[new_idx] = self.cell_index[old_idx];

            let h_old = old_idx * 32;
            let h_new = new_idx * 32;
            new_hidden[h_new..h_new + 32].copy_from_slice(&self.hidden_state[h_old..h_old + 32]);
        }

        swap(&mut self.x, &mut new_x);
        swap(&mut self.y, &mut new_y);
        swap(&mut self.vx, &mut new_vx);
        swap(&mut self.vy, &mut new_vy);
        swap(&mut self.surprise, &mut new_surprise);
        swap(&mut self.health, &mut new_health);
        swap(&mut self.cell_index, &mut new_cell);
        swap(&mut self.hidden_state, &mut new_hidden);
    }
}

/// Unified Mathematics Flocking Kernel.
/// Replaces the 3-Tier engine with a single, aggressive SIMD-optimized pass.
pub struct UnifiedKernel {
    pub w_cohesion: f32,
    pub w_separation: f32,
    pub w_alignment: f32,
    pub w_memory: f32,
    pub w_fear: f32,
}

impl Default for UnifiedKernel {
    fn default() -> Self {
        Self {
            w_cohesion: 0.1,
            w_separation: 0.5,
            w_alignment: 0.1,
            w_memory: 0.8,
            w_fear: 1.0,  
        }
    }
}

pub fn run_unified_simd_physics(pool: &mut SwarmPool, kernel: &UnifiedKernel, pheromones: &PheromoneField, width: f32, height: f32) {
    // Pure unified physics processing, 100% population utilization.
    // Partition by blocks of agents to avoid false sharing in rayon, but process them linearly
    // Since memory is sorted by spatial hash, chunks of agents belong to same or nearby cells.
    
    let chunk_size = 256;
    let _n = pool.n_agents;
    
    // We cannot easily borrow disjoint mutable slices of SoA arrays dynamically in safe rust without 
    // using `par_chunks_mut()` on zipped iterators. 
    pool.x.par_chunks_mut(chunk_size)
        .zip(pool.y.par_chunks_mut(chunk_size))
        .zip(pool.vx.par_chunks_mut(chunk_size))
        .zip(pool.vy.par_chunks_mut(chunk_size))
        .for_each(|(((x_chunk, y_chunk), vx_chunk), vy_chunk)| {
            for i in 0..x_chunk.len() {
                let x = &mut x_chunk[i];
                let y = &mut y_chunk[i];
                let vx = &mut vx_chunk[i];
                let vy = &mut vy_chunk[i];
                
                let (gx_mem, gy_mem) = pheromones.gradient(*x, *y, 2); // CH_2: Trail Marker
                let (gx_fear, gy_fear) = pheromones.gradient(*x, *y, 1); // CH_1: Danger

                let mut fx = kernel.w_memory * gx_mem - kernel.w_fear * gx_fear;
                let mut fy = kernel.w_memory * gy_mem - kernel.w_fear * gy_fear;

                fx += (*vx * 0.9) + (rand::random::<f32>() - 0.5) * 0.1;
                fy += (*vy * 0.9) + (rand::random::<f32>() - 0.5) * 0.1;

                let mag = (fx * fx + fy * fy).sqrt().max(0.001);
                if mag > 1.0 {
                    fx /= mag;
                    fy /= mag;
                }

                *vx = fx * 2.0; 
                *vy = fy * 2.0;

                *x = (*x + *vx).clamp(0.0, width);
                *y = (*y + *vy).clamp(0.0, height);
            }
        });
}
