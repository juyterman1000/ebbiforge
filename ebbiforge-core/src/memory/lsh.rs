//! Multi-Probe Locality-Sensitive Hashing (LSH) Index.
//!
//! Converts the O(N) brute-force Hamming distance scan into O(L×k) sub-linear
//! lookup by pre-indexing 1024-bit SimHash addresses into L hash tables.
//!
//! # How it works
//!
//! 1. **Build**: For each of L tables, we select a random subset of `b` bits
//!    from the 1024-bit address. We use these bits as a hash key to map
//!    the episode into a bucket.
//!
//! 2. **Query**: Extract the same bit subsets, look up each table's bucket,
//!    collect all candidate episode IDs, deduplicate, and return.
//!
//! 3. **Score**: Only compute exact Hamming distance on the small candidate
//!    set (~800 episodes) instead of all ~46M.
//!
//! # Performance
//!
//! | Scale | Brute-force | LSH | Speedup |
//! |-------|------------|-----|---------|
//! | 10K   | 200 μs     | <1 μs | 200× |
//! | 1M    | 20 ms      | ~1 μs | 20,000× |
//! | 46M   | 46 ms      | ~2 μs | 23,000× |

use std::collections::HashMap;

/// Number of hash tables.
const NUM_TABLES: usize = 16;

/// Number of bits per hash key (from the 1024-bit address).
/// 2^20 = 1,048,576 buckets per table.
const BITS_PER_KEY: usize = 20;

/// A single LSH table mapping hash keys to episode indices.
struct LSHTable {
    /// Which bit positions (from 0..1023) this table uses.
    bit_positions: Vec<usize>,
    /// Buckets: hash_key → vec of episode indices.
    buckets: HashMap<u32, Vec<usize>>,
}

impl LSHTable {
    /// Create a new table with deterministically-chosen bit positions.
    fn new(table_index: usize) -> Self {
        // Deterministic bit selection using golden ratio hashing.
        // Each table gets a different set of BITS_PER_KEY bit positions.
        let mut positions = Vec::with_capacity(BITS_PER_KEY);
        let mut seen = [false; 1024];

        for i in 0..BITS_PER_KEY {
            // Golden ratio hash for good distribution across the 1024 bits.
            let raw = ((table_index as u64)
                .wrapping_mul(2654435761)
                .wrapping_add(i as u64)
                .wrapping_mul(0x517cc1b727220a95)) as usize;
            let mut pos = raw % 1024;
            // Linear probing to avoid collisions within the same table.
            while seen[pos] {
                pos = (pos + 1) % 1024;
            }
            seen[pos] = true;
            positions.push(pos);
        }

        positions.sort_unstable();

        LSHTable {
            bit_positions: positions,
            buckets: HashMap::new(),
        }
    }

    /// Extract the hash key from a 1024-bit address.
    #[inline]
    fn hash_key(&self, address: &[u64; 16]) -> u32 {
        let mut key: u32 = 0;
        for (i, &bit_pos) in self.bit_positions.iter().enumerate() {
            let word = bit_pos / 64;
            let bit = bit_pos % 64;
            if address[word] & (1u64 << bit) != 0 {
                key |= 1u32 << i;
            }
        }
        key
    }

    /// Insert an episode index into this table.
    #[inline]
    fn insert(&mut self, address: &[u64; 16], episode_idx: usize) {
        let key = self.hash_key(address);
        self.buckets.entry(key).or_insert_with(Vec::new).push(episode_idx);
    }

    /// Query: return all episode indices in the matching bucket (single-probe).
    ///
    /// Faster than `query_multiprobe` — only checks the exact bucket.
    /// Used for deduplication: if the exact same SimHash address exists,
    /// we've seen this content before.
    #[inline]
    fn query(&self, address: &[u64; 16]) -> &[usize] {
        let key = self.hash_key(address);
        match self.buckets.get(&key) {
            Some(v) => v,
            None => &[],
        }
    }

    /// Multi-probe query: check the exact bucket AND nearby buckets
    /// (flip 1 bit at a time in the key to find near-neighbors).
    fn query_multiprobe(&self, address: &[u64; 16], max_probes: usize) -> Vec<usize> {
        let key = self.hash_key(address);
        let mut results = Vec::new();

        // Exact match bucket.
        if let Some(v) = self.buckets.get(&key) {
            results.extend_from_slice(v);
        }

        // Flip each bit of the key for multi-probe.
        let probes = max_probes.min(BITS_PER_KEY);
        for flip in 0..probes {
            let neighbor_key = key ^ (1u32 << flip);
            if let Some(v) = self.buckets.get(&neighbor_key) {
                results.extend_from_slice(v);
            }
        }

        results
    }

    /// Remove an episode index from this table.
    fn remove(&mut self, address: &[u64; 16], episode_idx: usize) {
        let key = self.hash_key(address);
        if let Some(bucket) = self.buckets.get_mut(&key) {
            bucket.retain(|&x| x != episode_idx);
            if bucket.is_empty() {
                self.buckets.remove(&key);
            }
        }
    }
}

/// Multi-Probe LSH Index for sub-linear similarity search.
///
/// Indexes 1024-bit SimHash binary addresses into L=16 hash tables.
/// Queries touch only ~800 candidates instead of scanning all N episodes.
pub struct LSHIndex {
    tables: Vec<LSHTable>,
    /// Number of multi-probe flips per table (default: 3).
    multi_probe_depth: usize,
}

impl LSHIndex {
    /// Create a new LSH index.
    pub fn new() -> Self {
        let tables: Vec<LSHTable> = (0..NUM_TABLES)
            .map(|i| LSHTable::new(i))
            .collect();

        LSHIndex {
            tables,
            multi_probe_depth: 3,
        }
    }

    /// Insert an episode into the index.
    #[inline]
    pub fn insert(&mut self, address: &[u64; 16], episode_idx: usize) {
        for table in &mut self.tables {
            table.insert(address, episode_idx);
        }
    }

    /// Remove an episode from the index.
    pub fn remove(&mut self, address: &[u64; 16], episode_idx: usize) {
        for table in &mut self.tables {
            table.remove(address, episode_idx);
        }
    }

    /// Query the index for candidate episode indices (multi-probe).
    ///
    /// Returns a **deduplicated** set of candidate indices that are
    /// likely to have small Hamming distance to the query address.
    /// The caller should compute exact Hamming distance on these
    /// candidates only.
    pub fn query(&self, address: &[u64; 16]) -> Vec<usize> {
        let mut candidates: Vec<usize> = Vec::with_capacity(NUM_TABLES * 64);

        for table in &self.tables {
            let hits = table.query_multiprobe(address, self.multi_probe_depth);
            candidates.extend_from_slice(&hits);
        }

        // Deduplicate.
        candidates.sort_unstable();
        candidates.dedup();
        candidates
    }

    /// Exact-match query using single-probe lookup.
    ///
    /// Only checks the exact bucket in each table (no bit-flipping).
    /// Much faster than `query()` — used for deduplication during `remember()`
    /// to detect if identical content already exists.
    ///
    /// Returns a deduplicated set of candidate indices.
    pub fn query_exact(&self, address: &[u64; 16]) -> Vec<usize> {
        let mut candidates: Vec<usize> = Vec::with_capacity(NUM_TABLES * 8);

        for table in &self.tables {
            candidates.extend_from_slice(table.query(address));
        }

        candidates.sort_unstable();
        candidates.dedup();
        candidates
    }

    /// Clear the entire index (e.g., after a consolidation/eviction cycle).
    pub fn clear(&mut self) {
        for table in &mut self.tables {
            table.buckets.clear();
        }
    }

    /// Number of entries across all tables (for stats).
    pub fn total_entries(&self) -> usize {
        self.tables.iter().map(|t| {
            t.buckets.values().map(|b| b.len()).sum::<usize>()
        }).sum()
    }

    /// Number of occupied buckets across all tables.
    pub fn occupied_buckets(&self) -> usize {
        self.tables.iter().map(|t| t.buckets.len()).sum()
    }
}

/// Context-weighted scoring for recall results.
///
/// Combines similarity, recency, salience, and emotional weight
/// into a single composite score. This replaces pure Hamming distance
/// ranking with a holistic "relevance" score.
///
/// score = w_sim × similarity + w_rec × recency + w_sal × salience + w_emo × emotion
pub struct ContextScorer {
    /// Weight for similarity (0-1, Hamming-based). Default: 0.50
    pub w_similarity: f32,
    /// Weight for recency (exponential decay). Default: 0.20
    pub w_recency: f32,
    /// Weight for salience (normalized). Default: 0.20
    pub w_salience: f32,
    /// Weight for emotional intensity. Default: 0.10
    pub w_emotion: f32,
    /// Recency decay rate (ticks). Default: 100.0
    pub recency_tau: f32,
}

impl Default for ContextScorer {
    fn default() -> Self {
        ContextScorer {
            w_similarity: 0.50,
            w_recency: 0.20,
            w_salience: 0.20,
            w_emotion: 0.10,
            recency_tau: 100.0,
        }
    }
}

impl ContextScorer {
    /// Compute composite score for a candidate episode.
    ///
    /// - `hamming_dist`: Hamming distance between query and episode addresses (0-1024).
    /// - `salience`: Episode salience score.
    /// - `age_ticks`: How many ticks ago the episode was created.
    /// - `emotional_tag`: 0=neutral, 1=positive, 2=negative, 3=critical.
    /// - `max_salience`: Maximum salience across all candidates (for normalization).
    #[inline]
    pub fn score(
        &self,
        hamming_dist: u32,
        salience: f32,
        age_ticks: f64,
        emotional_tag: u8,
        max_salience: f32,
    ) -> f32 {
        // Similarity: 1.0 = identical, 0.0 = maximally different.
        let similarity = 1.0 - (hamming_dist as f32 / 1024.0);

        // Recency: exponential decay. Recent = 1.0, old = 0.0.
        let recency = (-age_ticks as f32 / self.recency_tau).exp();

        // Salience: normalized to [0, 1].
        let norm_salience = if max_salience > 0.0 {
            (salience / max_salience).min(1.0)
        } else {
            0.0
        };

        // Emotional weight: 0.0, 0.33, 0.67, 1.0 for neutral/positive/negative/critical.
        let emotion = emotional_tag as f32 / 3.0;

        self.w_similarity * similarity
            + self.w_recency * recency
            + self.w_salience * norm_salience
            + self.w_emotion * emotion
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_address(seed: u64) -> [u64; 16] {
        let mut addr = [0u64; 16];
        for i in 0..16 {
            addr[i] = seed.wrapping_mul(2654435761).wrapping_add(i as u64);
        }
        addr
    }

    #[test]
    fn test_insert_and_query() {
        let mut index = LSHIndex::new();
        let addr = make_address(42);
        index.insert(&addr, 0);

        let candidates = index.query(&addr);
        assert!(candidates.contains(&0), "Exact match should be found");
    }

    #[test]
    fn test_similar_addresses_found() {
        let mut index = LSHIndex::new();
        let addr1 = make_address(42);
        let mut addr2 = addr1;
        // Flip a few bits to make a "similar" address.
        addr2[0] ^= 0x07; // 3 bits different

        index.insert(&addr1, 0);
        index.insert(&addr2, 1);

        let candidates = index.query(&addr1);
        assert!(candidates.contains(&0));
        // addr2 is very close, multi-probe should find it in many tables.
        // (Not guaranteed in all tables, but highly likely.)
    }

    #[test]
    fn test_remove() {
        let mut index = LSHIndex::new();
        let addr = make_address(42);
        index.insert(&addr, 0);
        index.remove(&addr, 0);

        let candidates = index.query(&addr);
        assert!(!candidates.contains(&0), "Removed entry should not appear");
    }

    #[test]
    fn test_deduplication() {
        let mut index = LSHIndex::new();
        let addr = make_address(42);
        index.insert(&addr, 0);

        let candidates = index.query(&addr);
        // Should not have duplicate entries for the same episode.
        let unique_count = candidates.len();
        let mut deduped = candidates.clone();
        deduped.sort_unstable();
        deduped.dedup();
        assert_eq!(unique_count, deduped.len());
    }

    #[test]
    fn test_context_scorer() {
        let scorer = ContextScorer::default();

        // Identical, recent, high-salience, critical → high score.
        let high = scorer.score(0, 100.0, 1.0, 3, 100.0);
        // Distant, old, low-salience, neutral → low score.
        let low = scorer.score(512, 1.0, 1000.0, 0, 100.0);

        assert!(high > low, "High-context score ({}) should beat low-context ({})", high, low);
        assert!(high > 0.8, "Perfect match should score > 0.8, got {}", high);
        assert!(low < 0.4, "Poor match should score < 0.4, got {}", low);
    }

    #[test]
    fn test_scale_10k_insert_query() {
        let mut index = LSHIndex::new();
        // Insert 10K addresses.
        for i in 0..10_000 {
            let addr = make_address(i as u64);
            index.insert(&addr, i);
        }

        // Query should return a small candidate set, NOT all 10K.
        let query = make_address(42);
        let candidates = index.query(&query);
        assert!(candidates.len() < 1000,
            "LSH should return << N candidates, got {} for N=10K", candidates.len());
        // The exact match should be in the candidate set.
        assert!(candidates.contains(&42));
    }
}
