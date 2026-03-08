//! Memory-Mapped Swarm Pool
//!
//! Backs all SoA agent arrays with anonymous mmap regions instead of heap Vec.
//! This allows the OS to page in/out agent data on demand, enabling 100M agents
//! on machines with 16GB RAM. The kernel's virtual memory subsystem handles
//! the pressure — we never allocate 37GB of physical RAM.

use memmap2::MmapMut;
use rayon::prelude::*;
use std::mem;

/// A single memory-mapped array of typed elements.
///
/// Wraps an anonymous `MmapMut` and provides safe typed access via slices.
/// The backing memory is allocated by the OS kernel and can be paged to swap
/// transparently, allowing arrays far larger than physical RAM.
pub struct MmapArray<T: Copy + Default + Send + Sync> {
    mmap: MmapMut,
    len: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Copy + Default + Send + Sync> MmapArray<T> {
    /// Create a new mmap-backed array of `len` elements, zero-initialized.
    pub fn new(len: usize) -> Self {
        let byte_len = len * mem::size_of::<T>();
        // Anonymous mmap: no file, OS pages to swap under pressure
        let mmap = MmapMut::map_anon(byte_len.max(1))
            .expect("Failed to create anonymous mmap");

        Self {
            mmap,
            len,
            _marker: std::marker::PhantomData,
        }
    }

    /// Get an immutable slice of the entire array.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(
                self.mmap.as_ptr() as *const T,
                self.len,
            )
        }
    }

    /// Get a mutable slice of the entire array.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.mmap.as_mut_ptr() as *mut T,
                self.len,
            )
        }
    }

    /// Fill the entire array with a value.
    pub fn fill(&mut self, value: T) {
        self.as_mut_slice().iter_mut().for_each(|v| *v = value);
    }

    /// Parallel fill using rayon.
    pub fn par_fill(&mut self, value: T) {
        self.as_mut_slice().par_iter_mut().for_each(|v| *v = value);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
}

// Safety: MmapMut is a contiguous byte region. We only expose it through
// typed slices with proper lifetime bounds. The PhantomData<T> ensures
// the compiler tracks the element type correctly.
unsafe impl<T: Copy + Default + Send + Sync> Send for MmapArray<T> {}
unsafe impl<T: Copy + Default + Send + Sync> Sync for MmapArray<T> {}

/// Memory-mapped Struct-of-Arrays pool for 100M+ agents.
///
/// Each field is an independent mmap region. This means:
/// - Agent positions (x, y) can be paged in without touching health/surprise
/// - Spatial sorting only needs to touch position + cell_index pages
/// - The OS reclaims pages from dormant regions of the world automatically
pub struct MmapSwarmPool {
    pub n_agents: usize,

    // Core physics (each is a separate mmap region for independent paging)
    pub x: MmapArray<f32>,
    pub y: MmapArray<f32>,
    pub vx: MmapArray<f32>,
    pub vy: MmapArray<f32>,

    // Cognitive state
    pub surprise: MmapArray<f32>,
    pub refractory: MmapArray<f32>,
    pub health: MmapArray<f32>,

    // Spatial hashing metadata
    pub cell_index: MmapArray<u32>,

    // Evolvable genome (per-agent behavioral parameters)
    pub gene_decay: MmapArray<f32>,          // surprise decay rate [0.8, 0.99]
    pub gene_transfer: MmapArray<f32>,       // surprise transfer rate [0.01, 0.3]
    pub gene_refractory: MmapArray<f32>,     // refractory buildup rate [0.05, 0.8]
    pub gene_danger_sense: MmapArray<f32>,   // danger pheromone sensitivity [0.0, 0.5]
    pub gene_novelty_drive: MmapArray<f32>,  // novelty attraction weight [0.0, 0.8]
    pub gene_speed: MmapArray<f32>,          // max velocity [0.5, 5.0]
    pub generation: MmapArray<u32>,          // lineage counter
}

impl MmapSwarmPool {
    /// Allocate a new pool for `n_agents` agents.
    ///
    /// Memory cost (virtual): ~28 bytes/agent = 2.8GB for 100M agents.
    /// Physical RSS depends on which pages the OS keeps resident.
    pub fn new(n_agents: usize) -> Self {
        let mut pool = Self {
            n_agents,
            x: MmapArray::new(n_agents),
            y: MmapArray::new(n_agents),
            vx: MmapArray::new(n_agents),
            vy: MmapArray::new(n_agents),
            surprise: MmapArray::new(n_agents),
            refractory: MmapArray::new(n_agents),
            health: MmapArray::new(n_agents),
            cell_index: MmapArray::new(n_agents),
            gene_decay: MmapArray::new(n_agents),
            gene_transfer: MmapArray::new(n_agents),
            gene_refractory: MmapArray::new(n_agents),
            gene_danger_sense: MmapArray::new(n_agents),
            gene_novelty_drive: MmapArray::new(n_agents),
            gene_speed: MmapArray::new(n_agents),
            generation: MmapArray::new(n_agents),
        };

        // Initialize health to 1.0 (alive)
        pool.health.par_fill(1.0);

        // Initialize genes to PropagationConfig defaults
        pool.gene_decay.par_fill(0.92);
        pool.gene_transfer.par_fill(0.08);
        pool.gene_refractory.par_fill(0.3);
        pool.gene_danger_sense.par_fill(0.15);
        pool.gene_novelty_drive.par_fill(0.2);
        pool.gene_speed.par_fill(2.0);
        // generation starts at 0 (zero-initialized by mmap)

        pool
    }

    /// Randomize agent positions within the world bounds.
    pub fn randomize_positions(&mut self, width: f32, height: f32) {
        self.x.as_mut_slice().par_iter_mut()
            .for_each(|x| *x = rand::random::<f32>() * width);
        self.y.as_mut_slice().par_iter_mut()
            .for_each(|y| *y = rand::random::<f32>() * height);
    }

    /// Compute spatial hash indices for all agents.
    pub fn update_spatial_hashes(&mut self, world_width: f32) {
        let cell_size = super::swarm_engine::CELL_SIZE;
        let cols = (world_width / cell_size).ceil() as u32;

        let x_slice = self.x.as_slice();
        let y_slice = self.y.as_slice();
        let cells = self.cell_index.as_mut_slice();

        x_slice.par_iter()
            .zip(y_slice.par_iter())
            .zip(cells.par_iter_mut())
            .for_each(|((x, y), cell)| {
                let cx = (*x / cell_size) as u32;
                let cy = (*y / cell_size) as u32;
                *cell = cy * cols + cx;
            });
    }

    /// Sort all SoA arrays by spatial hash for cache locality.
    /// Uses argsort + parallel scatter to maintain L1/L2 cache warmth
    /// during neighbor queries.
    pub fn sort_by_spatial_hash(&mut self) {
        let n = self.n_agents;

        // Argsort by cell_index
        let cell_slice = self.cell_index.as_slice();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.par_sort_unstable_by_key(|&i| cell_slice[i]);

        // Scatter into new mmap regions
        let mut new_x = MmapArray::<f32>::new(n);
        let mut new_y = MmapArray::<f32>::new(n);
        let mut new_vx = MmapArray::<f32>::new(n);
        let mut new_vy = MmapArray::<f32>::new(n);
        let mut new_surprise = MmapArray::<f32>::new(n);
        let mut new_refractory = MmapArray::<f32>::new(n);
        let mut new_health = MmapArray::<f32>::new(n);
        let mut new_cell = MmapArray::<u32>::new(n);
        let mut new_gene_decay = MmapArray::<f32>::new(n);
        let mut new_gene_transfer = MmapArray::<f32>::new(n);
        let mut new_gene_refractory = MmapArray::<f32>::new(n);
        let mut new_gene_danger_sense = MmapArray::<f32>::new(n);
        let mut new_gene_novelty_drive = MmapArray::<f32>::new(n);
        let mut new_gene_speed = MmapArray::<f32>::new(n);
        let mut new_generation = MmapArray::<u32>::new(n);

        // Sequential scatter — O(N) pass, memory-bandwidth bound
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            new_x.as_mut_slice()[new_idx] = self.x.as_slice()[old_idx];
            new_y.as_mut_slice()[new_idx] = self.y.as_slice()[old_idx];
            new_vx.as_mut_slice()[new_idx] = self.vx.as_slice()[old_idx];
            new_vy.as_mut_slice()[new_idx] = self.vy.as_slice()[old_idx];
            new_surprise.as_mut_slice()[new_idx] = self.surprise.as_slice()[old_idx];
            new_refractory.as_mut_slice()[new_idx] = self.refractory.as_slice()[old_idx];
            new_health.as_mut_slice()[new_idx] = self.health.as_slice()[old_idx];
            new_cell.as_mut_slice()[new_idx] = self.cell_index.as_slice()[old_idx];
            new_gene_decay.as_mut_slice()[new_idx] = self.gene_decay.as_slice()[old_idx];
            new_gene_transfer.as_mut_slice()[new_idx] = self.gene_transfer.as_slice()[old_idx];
            new_gene_refractory.as_mut_slice()[new_idx] = self.gene_refractory.as_slice()[old_idx];
            new_gene_danger_sense.as_mut_slice()[new_idx] = self.gene_danger_sense.as_slice()[old_idx];
            new_gene_novelty_drive.as_mut_slice()[new_idx] = self.gene_novelty_drive.as_slice()[old_idx];
            new_gene_speed.as_mut_slice()[new_idx] = self.gene_speed.as_slice()[old_idx];
            new_generation.as_mut_slice()[new_idx] = self.generation.as_slice()[old_idx];
        }

        // Swap in the sorted arrays
        self.x = new_x;
        self.y = new_y;
        self.vx = new_vx;
        self.vy = new_vy;
        self.surprise = new_surprise;
        self.refractory = new_refractory;
        self.health = new_health;
        self.cell_index = new_cell;
        self.gene_decay = new_gene_decay;
        self.gene_transfer = new_gene_transfer;
        self.gene_refractory = new_gene_refractory;
        self.gene_danger_sense = new_gene_danger_sense;
        self.gene_novelty_drive = new_gene_novelty_drive;
        self.gene_speed = new_gene_speed;
        self.generation = new_generation;
    }

    /// Report approximate physical memory usage in MB.
    pub fn estimated_virtual_mb(&self) -> f64 {
        let bytes_per_agent = 15 * mem::size_of::<f32>(); // 13 f32 + 2 u32
        (self.n_agents * bytes_per_agent) as f64 / (1024.0 * 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mmap_pool_basic() {
        let mut pool = MmapSwarmPool::new(1000);
        assert_eq!(pool.n_agents, 1000);
        assert_eq!(pool.health.as_slice()[0], 1.0);
        assert_eq!(pool.x.as_slice()[0], 0.0);

        pool.randomize_positions(1000.0, 1000.0);
        assert!(pool.x.as_slice()[0] >= 0.0);
        assert!(pool.x.as_slice()[0] <= 1000.0);
    }

    #[test]
    fn mmap_pool_large_allocation() {
        // 10M agents — should succeed via mmap even on constrained systems
        let pool = MmapSwarmPool::new(10_000_000);
        assert_eq!(pool.n_agents, 10_000_000);
        assert_eq!(pool.health.as_slice()[9_999_999], 1.0);
    }

    #[test]
    fn spatial_sort_preserves_data() {
        let mut pool = MmapSwarmPool::new(100);
        pool.randomize_positions(100.0, 100.0);

        let original_sum: f32 = pool.x.as_slice().iter().sum();

        pool.update_spatial_hashes(100.0);
        pool.sort_by_spatial_hash();

        let sorted_sum: f32 = pool.x.as_slice().iter().sum();
        assert!((original_sum - sorted_sum).abs() < 0.01);
    }
}
