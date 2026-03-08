use std::f32;

/// Multi-channel pheromone field.
/// Each channel represents a different "chemical" for emergent stigmergic coordination.
/// CH_0: Resource Abundance
/// CH_1: Danger Signal
/// CH_2: Trail Marker
/// CH_3: Hoarding Suppressor
/// CH_4: Novelty Beacon
/// CH_5: Alliance Signal
pub struct PheromoneField {
    pub data: Box<[f32]>,
    pub channels: usize,
    pub width: usize,
    pub height: usize,
    pub cell_size: f32,
    pub decay_rates: [f32; 6],
    pub diffusion: [f32; 6],
}

impl PheromoneField {
    pub fn new(width: usize, height: usize, cell_size: f32) -> Self {
        let channels = 6;
        let total_cells = channels * width * height;
        Self {
            data: vec![0.0; total_cells].into_boxed_slice(),
            channels,
            width,
            height,
            cell_size,
            decay_rates: [0.005, 0.02, 0.003, 0.01, 0.015, 0.008],
            diffusion: [0.1, 0.3, 0.05, 0.2, 0.25, 0.1],
        }
    }

    /// Convert real-world coordinates to grid indices with fractional bounds for bilinear sampling
    fn bilinear_coords(&self, x: f32, y: f32) -> (usize, usize, f32, f32) {
        let gx = (x / self.cell_size).clamp(0.0, (self.width - 2) as f32);
        let gy = (y / self.cell_size).clamp(0.0, (self.height - 2) as f32);
        
        let cx = gx.floor() as usize;
        let cy = gy.floor() as usize;
        
        (cx, cy, gx - cx as f32, gy - cy as f32)
    }

    pub fn deposit(&mut self, x: f32, y: f32, channel: usize, amount: f32) {
        if channel >= self.channels { return; }
        let (cx, cy, fx, fy) = self.bilinear_coords(x, y);
        let w = self.width;
        let ch_off = channel * w * self.height;

        // Bilinear splatting across 4 cells
        self.data[ch_off + cy * w + cx] += amount * (1.0 - fx) * (1.0 - fy);
        self.data[ch_off + cy * w + (cx + 1)] += amount * fx * (1.0 - fy);
        self.data[ch_off + (cy + 1) * w + cx] += amount * (1.0 - fx) * fy;
        self.data[ch_off + (cy + 1) * w + (cx + 1)] += amount * fx * fy;
    }

    pub fn sample(&self, x: f32, y: f32, channel: usize) -> f32 {
        if channel >= self.channels { return 0.0; }
        let (cx, cy, _, _) = self.bilinear_coords(x, y);
        let ch_off = channel * self.width * self.height;
        // Nearest neighbor approximation for gradient fetching speed
        self.data[ch_off + cy * self.width + cx]
    }

    /// Retrieve the gradient (dx, dy) to steer agents based on pheromone density
    pub fn gradient(&self, x: f32, y: f32, channel: usize) -> (f32, f32) {
        let eps = self.cell_size;
        let cx = self.sample(x + eps, y, channel) - self.sample(x - eps, y, channel);
        let cy = self.sample(x, y + eps, channel) - self.sample(x, y - eps, channel);
        (cx / (2.0 * eps), cy / (2.0 * eps))
    }

    /// AVX2-friendly Finite Difference Heat Equation
    pub fn tick(&mut self) {
        let w = self.width;
        let h = self.height;
        
        // We need a scratch buffer for synchronous updates to prevent directional bias
        let mut next_data = self.data.clone();

        for ch in 0..self.channels {
            let rate = self.decay_rates[ch];
            let d = self.diffusion[ch];
            let off = ch * w * h;

            for i in 1..h-1 {
                for j in 1..w-1 {
                    let idx = off + i * w + j;
                    
                    // 5-point stencil Laplacian
                    let laplacian = self.data[off + (i - 1) * w + j] 
                                  + self.data[off + (i + 1) * w + j]
                                  + self.data[off + i * w + (j - 1)] 
                                  + self.data[off + i * w + (j + 1)]
                                  - 4.0 * self.data[idx];
                                  
                    // Diffusion + Decay
                    next_data[idx] = (self.data[idx] + d * laplacian) * (1.0 - rate);
                }
            }
        }
        
        self.data = next_data;
    }
}
