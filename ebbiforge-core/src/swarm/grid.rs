// grid.rs — Spatial hash grid.
//
// Cell size = perception radius → agents only need to check 3×3 = 9 cells.
// Hash: Fibonacci hashing (Knuth).  Near-zero collision rate for spatial keys.
// Rebuild: one O(N) pass, no sort, no tree.

use super::swarm_engine::SwarmPool;

/// Spatial hash grid.  Rebuild every tick before neighborhood queries.
pub struct SpatialHashGrid {
    /// Flattened bucket list: each bucket is a contiguous slice in `data`.
    /// `head[h]`   = start index in `data` for bucket h.
    /// `count[h]`  = number of agents in bucket h.
    ///
    /// We use a two-pass approach:
    ///   Pass 1: count agents per bucket.
    ///   Pass 2: scatter agents into pre-allocated runs.
    /// This avoids Vec-of-Vec heap fragmentation entirely.
    counts: Vec<u32>,         // [table_size]  agents per bucket
    offsets: Vec<u32>,        // [table_size]  start of each bucket in `data`
    data: Vec<u32>,           // [N]           agent indices, packed
    table_size: usize,        // power of two
    mask: usize,              // table_size - 1
    pub cell_size: f32,
    pub world_min: [f32; 2],
}

impl SpatialHashGrid {
    /// `table_size` should be ≥ 2× expected agent count for low collision rate.
    pub fn new(table_size: usize, cell_size: f32, world_min: [f32; 2]) -> Self {
        assert!(table_size.is_power_of_two(), "table_size must be a power of two");
        SpatialHashGrid {
            counts:  vec![0u32; table_size],
            offsets: vec![0u32; table_size],
            data:    Vec::new(),
            table_size,
            mask: table_size - 1,
            cell_size,
            world_min,
        }
    }

    /// Fibonacci hash for (cx, cy) cell coordinates.
    /// Knuth multiplicative hashing — uniform distribution for spatial keys.
    #[inline(always)]
    fn hash(&self, cx: i32, cy: i32) -> usize {
        // Combine two i32s into one u64, then Fibonacci hash
        let key = (cx as u64).wrapping_mul(2654435761)
                ^ (cy as u64).wrapping_mul(2246822519);
        // Fibonacci multiplier for u64: 11400714819323198485
        (key.wrapping_mul(11400714819323198485) >> (64 - self.table_size.trailing_zeros())) as usize
            & self.mask
    }

    #[inline(always)]
    pub fn world_to_cell(&self, x: f32, y: f32) -> (i32, i32) {
        let cx = ((x - self.world_min[0]) / self.cell_size).floor() as i32;
        let cy = ((y - self.world_min[1]) / self.cell_size).floor() as i32;
        (cx, cy)
    }

    /// Full O(N) rebuild from agent pool.  Two-pass (count then scatter).
    pub fn rebuild(&mut self, pool: &SwarmPool) {
        let n = pool.n_agents;

        // Ensure data buffer is large enough
        if self.data.len() < n {
            self.data.resize(n, 0);
        }

        // ── Pass 1: count ────────────────────────────────────────────────────
        self.counts.iter_mut().for_each(|c| *c = 0);

        for i in 0..n {
            let (cx, cy) = self.world_to_cell(pool.x[i], pool.y[i]);
            let h = self.hash(cx, cy);
            self.counts[h] += 1;
        }

        // ── Prefix sum → offsets ─────────────────────────────────────────────
        let mut running = 0u32;
        for h in 0..self.table_size {
            self.offsets[h] = running;
            running += self.counts[h];
        }

        // ── Pass 2: scatter ──────────────────────────────────────────────────
        self.counts.iter_mut().for_each(|c| *c = 0);  // reuse as cursor

        for i in 0..n {
            let (cx, cy) = self.world_to_cell(pool.x[i], pool.y[i]);
            let h    = self.hash(cx, cy);
            let slot = (self.offsets[h] + self.counts[h]) as usize;
            self.data[slot] = i as u32;
            self.counts[h] += 1;
        }
    }

    // ── Decomposed rebuild API (for MmapSwarmPool integration) ────────────

    /// Reset all bucket counts to zero. Call before count_agent loop.
    pub fn counts_reset(&mut self) {
        self.counts.iter_mut().for_each(|c| *c = 0);
    }

    /// Increment the count for the bucket containing cell (cx, cy).
    #[inline]
    pub fn count_agent(&mut self, cx: i32, cy: i32) {
        let h = self.hash(cx, cy);
        self.counts[h] += 1;
    }

    /// Compute prefix-sum offsets from counts. Call after all count_agent calls.
    pub fn compute_offsets(&mut self) {
        // Ensure data buffer is large enough
        let total: u32 = self.counts.iter().sum();
        if self.data.len() < total as usize {
            self.data.resize(total as usize, 0);
        }

        let mut running = 0u32;
        for h in 0..self.table_size {
            self.offsets[h] = running;
            running += self.counts[h];
        }
        // Reset counts for scatter pass
        self.counts.iter_mut().for_each(|c| *c = 0);
    }

    /// Scatter agent index into the appropriate bucket. Call after compute_offsets.
    #[inline]
    pub fn scatter_agent(&mut self, cx: i32, cy: i32, agent_idx: u32) {
        let h = self.hash(cx, cy);
        let slot = (self.offsets[h] + self.counts[h]) as usize;
        self.data[slot] = agent_idx;
        self.counts[h] += 1;
    }


    /// Query all candidate neighbors within radius `r` of (qx, qy).
    ///
    /// Calls `callback(agent_idx)` for each candidate.
    /// Callers MUST still perform exact distance check — the hash grid
    /// is a filter, not a guarantee.
    ///
    /// Returns early if `callback` returns `false` (optional early exit).
    #[inline]
    pub fn query_radius<F>(&self, qx: f32, qy: f32, r: f32, mut callback: F)
    where
        F: FnMut(u32),
    {
        let (cx0, cy0) = self.world_to_cell(qx - r, qy - r);
        let (cx1, cy1) = self.world_to_cell(qx + r, qy + r);

        for cy in cy0..=cy1 {
            for cx in cx0..=cx1 {
                let h     = self.hash(cx, cy);
                let start = self.offsets[h] as usize;
                let end   = start + self.counts[h] as usize;
                for &idx in &self.data[start..end] {
                    callback(idx);
                }
            }
        }
    }

    /// Same as `query_radius` but skips `self_idx`.
    #[inline]
    pub fn query_neighbors<F>(
        &self,
        self_idx: u32,
        qx: f32, qy: f32, r: f32,
        mut callback: F,
    ) where
        F: FnMut(u32),
    {
        self.query_radius(qx, qy, r, |idx| {
            if idx != self_idx { callback(idx) }
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swarm::swarm_engine::SwarmPool;

    fn make_pool_circle(n: usize, radius: f32) -> SwarmPool {
        let mut pool = SwarmPool::new(n + 1);
        for i in 0..n {
            let angle = (i as f32 / n as f32) * std::f32::consts::TAU;
            pool.x[i] = radius * angle.cos();
            pool.y[i] = radius * angle.sin();
        }
        pool
    }

    #[test]
    fn rebuild_query_counts_match() {
        let pool  = make_pool_circle(200, 5.0);
        let mut grid = SpatialHashGrid::new(1024, 2.0, [-20.0, -20.0]);
        grid.rebuild(&pool);

        // Spatial hash grids are conservative filters — collisions may return extras
        let mut found = 0usize;
        grid.query_radius(0.0, 0.0, 8.0, |_| found += 1);
        assert!(found >= 200, "expected at least all agents in radius, got {}", found);
    }

    #[test]
    fn no_self_in_neighbors() {
        let pool = make_pool_circle(50, 3.0);
        let mut grid = SpatialHashGrid::new(256, 1.0, [-10.0, -10.0]);
        grid.rebuild(&pool);

        for i in 0..50u32 {
            grid.query_neighbors(i, pool.x[i as usize], pool.y[i as usize], 5.0,
                |j| assert_ne!(j, i));
        }
    }
}
