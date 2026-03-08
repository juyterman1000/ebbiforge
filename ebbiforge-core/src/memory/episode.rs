//! Episode — a single memory trace with Ebbinghaus-governed retention.
//!
//! Each episode carries content, a quantised embedding for similarity search,
//! and a salience score that controls how long it survives.

use pyo3::prelude::*;

/// Emotional tag controlling base salience multiplier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EmotionalTag {
    Neutral  = 0,
    Positive = 1,
    Negative = 2,
    Critical = 3,
}

impl From<u8> for EmotionalTag {
    fn from(v: u8) -> Self {
        match v {
            1 => EmotionalTag::Positive,
            2 => EmotionalTag::Negative,
            3 => EmotionalTag::Critical,
            _ => EmotionalTag::Neutral,
        }
    }
}

/// A single memory trace in the hippocampal buffer.
///
/// Size budget: ~224 bytes per episode → 10 GB ≈ 44M episodes on disk.
#[derive(Clone, Debug)]
#[pyclass]
pub struct Episode {
    /// Unique monotonic id within an engine instance.
    #[pyo3(get)]
    pub id: u64,

    /// Compact content string (the actual memory).
    #[pyo3(get)]
    pub content: String,

    /// Optional provenance source (file, URL, agent name).
    #[pyo3(get)]
    pub source: String,

    /// Quantised embedding — 64 × u16 values (128 bytes).
    /// Stored as u16 to halve memory vs f32 while preserving ranking order.
    #[pyo3(get)]
    pub embedding: Vec<u16>,

    /// Binary hash of the embedding for Kanerva SDM addressing.
    /// 1024-bit = 16 × u64 words.  Computed via SimHash at write time.
    pub binary_address: [u64; 16],

    /// Salience score controlling Ebbinghaus retention.
    ///   Retention(age) = e^(-age / salience)
    /// Higher salience → longer survival.
    #[pyo3(get)]
    pub salience: f32,

    /// Tick at which this episode was created.
    #[pyo3(get)]
    pub created_at: f64,

    /// Tick at which this episode was last recalled.
    #[pyo3(get)]
    pub last_recalled: f64,

    /// Number of times this episode has been recalled.
    #[pyo3(get)]
    pub recall_count: u32,

    /// Emotional intensity tag.
    pub emotional_tag: EmotionalTag,

    /// Whether this episode has been consolidated to the neocortex (Kanerva SDM).
    #[pyo3(get)]
    pub consolidated: bool,

    /// Relationship edges — up to 4 linked episode IDs.
    /// Enables "follow the thread" queries: complaint → threat → audit.
    #[pyo3(get)]
    pub related_to: Vec<u64>,
}

impl Episode {
    /// Compute Ebbinghaus retention at a given tick.
    ///
    ///   R(t) = e^(-(current_tick - created_at) / salience)
    ///
    /// Returns a value in [0, 1].  0 = fully forgotten, 1 = perfect recall.
    #[inline]
    pub fn retention(&self, current_tick: f64) -> f32 {
        let age = (current_tick - self.created_at) as f32;
        if age <= 0.0 || self.salience <= 0.0 {
            return 1.0;
        }
        (-age / self.salience).exp()
    }

    /// Apply spaced-recall reinforcement.
    ///
    /// Each recall multiplies salience by `factor` (default 1.3).
    /// This implements the spacing effect from cognitive psychology:
    /// memories retrieved more often become harder to forget.
    #[inline]
    pub fn reinforce(&mut self, current_tick: f64, factor: f32) {
        self.recall_count += 1;
        self.last_recalled = current_tick;
        self.salience *= factor;
        // Cap salience to prevent unbounded growth.
        self.salience = self.salience.min(1_000.0);
    }
}

/// Convert a floating-point embedding vector into a 1024-bit binary address
/// using SimHash (random hyperplane LSH).
///
/// For deterministic reproducibility we use a fixed seed derived from
/// the vector dimensions.  The resulting binary address preserves
/// approximate cosine similarity: similar vectors → small Hamming distance.
pub fn simhash_embedding(embedding: &[u16]) -> [u64; 16] {
    let mut address = [0u64; 16];
    let dim = embedding.len();
    if dim == 0 {
        return address;
    }

    // Use the embedding values to construct 1024 random hyperplane projections.
    // Each bit in the address = sign of one projection.
    for bit_idx in 0..1024usize {
        let mut accumulator: i64 = 0;
        for (d, &val) in embedding.iter().enumerate() {
            // Deterministic weight for (bit_idx, d) via golden ratio hash.
            // Uses a simple hash: wrapping multiply + XOR with golden ratio.
            let seed = (bit_idx as u64)
                .wrapping_mul(2654435761)
                .wrapping_add(d as u64)
                .wrapping_mul(0x517cc1b727220a95);
            // Map seed to {-1, +1}.
            let weight: i64 = if seed & 1 == 0 { 1 } else { -1 };
            accumulator += weight * (val as i64);
        }
        if accumulator >= 0 {
            let word = bit_idx / 64;
            let bit = bit_idx % 64;
            address[word] |= 1u64 << bit;
        }
    }
    address
}

/// Hamming distance between two 1024-bit binary addresses.
///
/// Uses `count_ones()` which compiles to the hardware POPCNT instruction
/// on x86-64 (available since Nehalem, 2008).
#[inline]
pub fn hamming_distance(a: &[u64; 16], b: &[u64; 16]) -> u32 {
    let mut dist = 0u32;
    for i in 0..16 {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

/// Quantise an f32 embedding vector to u16 (preserving ranking order).
///
///   val_u16 = clamp((val_f32 + 1.0) / 2.0 * 65535, 0, 65535)
///
/// Assumes the input values are roughly in [-1, 1] (normalised embeddings).
pub fn quantise_embedding(float_vec: &[f32]) -> Vec<u16> {
    float_vec
        .iter()
        .map(|&v| {
            let scaled = ((v + 1.0) * 0.5 * 65535.0).round();
            scaled.clamp(0.0, 65535.0) as u16
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retention_decay() {
        let ep = Episode {
            id: 0,
            content: "test".to_string(),
            source: String::new(),
            embedding: vec![32768; 64],
            binary_address: [0; 16],
            salience: 1.0,
            created_at: 0.0,
            last_recalled: 0.0,
            recall_count: 0,
            emotional_tag: EmotionalTag::Neutral,
            consolidated: false,
            related_to: Vec::new(),
        };
        // At tick 0, retention = 1.0
        assert!((ep.retention(0.0) - 1.0).abs() < 0.001);
        // At tick 1 with salience 1.0 → e^-1 ≈ 0.368
        assert!((ep.retention(1.0) - 0.3679).abs() < 0.01);
        // At tick 5 → e^-5 ≈ 0.0067
        assert!(ep.retention(5.0) < 0.01);
    }

    #[test]
    fn test_reinforcement() {
        let mut ep = Episode {
            id: 0,
            content: "test".to_string(),
            source: String::new(),
            embedding: vec![32768; 64],
            binary_address: [0; 16],
            salience: 1.0,
            created_at: 0.0,
            last_recalled: 0.0,
            recall_count: 0,
            emotional_tag: EmotionalTag::Neutral,
            consolidated: false,
            related_to: Vec::new(),
        };
        ep.reinforce(5.0, 1.3);
        assert_eq!(ep.recall_count, 1);
        assert!((ep.salience - 1.3).abs() < 0.001);
        assert!((ep.last_recalled - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_hamming_identical() {
        let a = [0xFFu64; 16];
        assert_eq!(hamming_distance(&a, &a), 0);
    }

    #[test]
    fn test_hamming_opposite() {
        let a = [0u64; 16];
        let b = [u64::MAX; 16];
        assert_eq!(hamming_distance(&a, &b), 1024);
    }

    #[test]
    fn test_quantise() {
        let v = vec![0.0f32, 1.0, -1.0, 0.5];
        let q = quantise_embedding(&v);
        assert_eq!(q[0], 32768); // midpoint
        assert_eq!(q[1], 65535); // max
        assert_eq!(q[2], 0);     // min
        assert!(q[3] >= 49151 && q[3] <= 49152); // 0.75 * 65535 ≈ 49151-49152 (rounding)
    }
}
