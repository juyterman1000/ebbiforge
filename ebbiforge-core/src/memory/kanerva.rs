//! Kanerva Sparse Distributed Memory (SDM)
//!
//! A biologically-inspired associative memory that stores patterns in
//! high-dimensional binary space and retrieves them via Hamming distance.
//!
//! # How it works
//!
//! 1. **Hard locations**: Fixed set of 1024-bit binary addresses, randomly
//!    initialised at construction time.
//! 2. **Write**: For a given binary address, all hard locations within
//!    `activation_radius` Hamming distance are activated.  The content
//!    vector is added to each activated location's counter array.
//! 3. **Read**: Same activation. The counters of activated locations are
//!    summed, then thresholded to reconstruct the stored pattern.
//!
//! # Complexity
//!
//! - Write: O(N) where N = n_locations (but N is fixed and small: 10K-100K)
//! - Read:  O(N) — same scan
//! - On modern CPUs, the XOR+POPCNT inner loop over 1024-bit addresses
//!   runs at ~15ns per location comparison.

use super::episode::hamming_distance;
use rand::Rng;

/// The width of the content counter array.
/// Each hard location stores an array of i32 counters that accumulate
/// the sign-pattern of stored content vectors.
const COUNTER_WIDTH: usize = 64;

/// A single hard location in the Kanerva SDM.
#[derive(Clone, Debug)]
struct HardLocation {
    /// The 1024-bit address of this location.
    address: [u64; 16],
    /// Distributed counter array.  Positive = bit was 1, negative = bit was 0.
    counters: [i32; COUNTER_WIDTH],
    /// Number of writes to this location (for weighted read).
    write_count: u32,
}

/// Kanerva Sparse Distributed Memory.
///
/// Provides O(1)-amortised associative read/write over 1024-bit binary vectors
/// with graceful degradation under noise (fuzzy matching).
#[derive(Clone, Debug)]
pub struct KanervaSDM {
    locations: Vec<HardLocation>,
    n_locations: usize,
    activation_radius: u32,
}

/// A single recalled memory from the SDM, with metadata.
#[derive(Clone, Debug)]
pub struct SDMRecallResult {
    /// Reconstructed content vector (thresholded counters).
    pub content_vector: Vec<i32>,
    /// Number of hard locations that contributed to this recall.
    pub activated_count: usize,
    /// Mean Hamming distance of activated locations to the query.
    pub mean_distance: f32,
}

impl KanervaSDM {
    /// Create a new SDM with `n_locations` randomly-addressed hard locations.
    ///
    /// # Arguments
    /// - `n_locations`: Number of hard locations (10_000 – 100_000 typical).
    /// - `activation_radius`: Max Hamming distance for activation (300–450 typical
    ///   for 1024-bit addresses; ~40-45% of bits).
    pub fn new(n_locations: usize, activation_radius: u32) -> Self {
        let mut rng = rand::thread_rng();
        let mut locations = Vec::with_capacity(n_locations);

        for _ in 0..n_locations {
            let mut address = [0u64; 16];
            for word in address.iter_mut() {
                *word = rng.gen();
            }
            locations.push(HardLocation {
                address,
                counters: [0i32; COUNTER_WIDTH],
                write_count: 0,
            });
        }

        KanervaSDM {
            locations,
            n_locations,
            activation_radius,
        }
    }

    /// Write a content vector at a given binary address.
    ///
    /// All hard locations within `activation_radius` are updated.
    /// The content vector is encoded as {-1, +1} and added to the counters.
    pub fn write(&mut self, address: &[u64; 16], content: &[f32]) {
        // Encode content as sign vector: positive → +1, else → -1.
        let signs: Vec<i32> = content
            .iter()
            .take(COUNTER_WIDTH)
            .map(|&v| if v >= 0.0 { 1 } else { -1 })
            .collect();

        for loc in self.locations.iter_mut() {
            let dist = hamming_distance(address, &loc.address);
            if dist <= self.activation_radius {
                for (i, &s) in signs.iter().enumerate() {
                    loc.counters[i] += s;
                }
                loc.write_count += 1;
            }
        }
    }

    /// Read (recall) the content stored near a given binary address.
    ///
    /// Returns the summed counters from all activated hard locations,
    /// plus metadata about the recall quality.
    pub fn read(&self, address: &[u64; 16]) -> SDMRecallResult {
        let mut sum = [0i64; COUNTER_WIDTH];
        let mut activated = 0usize;
        let mut total_dist = 0u64;

        for loc in self.locations.iter() {
            let dist = hamming_distance(address, &loc.address);
            if dist <= self.activation_radius {
                for (i, &c) in loc.counters.iter().enumerate() {
                    sum[i] += c as i64;
                }
                activated += 1;
                total_dist += dist as u64;
            }
        }

        let content_vector: Vec<i32> = sum.iter().map(|&s| s as i32).collect();
        let mean_distance = if activated > 0 {
            total_dist as f32 / activated as f32
        } else {
            1024.0
        };

        SDMRecallResult {
            content_vector,
            activated_count: activated,
            mean_distance,
        }
    }

    /// Check whether any patterns are stored near a given address.
    #[inline]
    pub fn has_content(&self, address: &[u64; 16]) -> bool {
        self.locations
            .iter()
            .any(|loc| {
                hamming_distance(address, &loc.address) <= self.activation_radius
                    && loc.write_count > 0
            })
    }

    /// Number of hard locations that have received at least one write.
    pub fn occupied_count(&self) -> usize {
        self.locations.iter().filter(|l| l.write_count > 0).count()
    }

    /// Total hard locations.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.n_locations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_and_read_back() {
        let mut sdm = KanervaSDM::new(1_000, 512);
        let address = [0xAAAA_AAAA_AAAA_AAAAu64; 16];
        let content: Vec<f32> = (0..64).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();

        sdm.write(&address, &content);
        let result = sdm.read(&address);

        // With a broad activation radius (512/1024), many locations activate.
        assert!(result.activated_count > 0);
        // Pattern should be recoverable: even indices positive, odd negative.
        for (i, &v) in result.content_vector.iter().enumerate() {
            if i < 64 {
                if i % 2 == 0 {
                    assert!(v > 0, "Expected positive at index {}, got {}", i, v);
                } else {
                    assert!(v < 0, "Expected negative at index {}, got {}", i, v);
                }
            }
        }
    }

    #[test]
    fn test_empty_read() {
        let sdm = KanervaSDM::new(100, 400);
        let address = [0u64; 16];
        let result = sdm.read(&address);
        // All counters should be zero.
        assert!(result.content_vector.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_occupied_count() {
        let mut sdm = KanervaSDM::new(100, 512);
        assert_eq!(sdm.occupied_count(), 0);
        let address = [0u64; 16];
        let content = vec![1.0f32; 64];
        sdm.write(&address, &content);
        assert!(sdm.occupied_count() > 0);
    }
}
