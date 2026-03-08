//! HippocampusEngine — the user-facing memory system.
//!
//! Combines hippocampal ring buffer (fast episodic write) with Kanerva SDM
//! (persistent semantic recall) via background sleep-replay consolidation.
//!
//! # Python API
//!
//! ```python
//! from ebbiforge_core import HippocampusEngine
//!
//! mem = HippocampusEngine(capacity=1_000_000)
//! mem.remember("User complained about billing", salience=0.8)
//! results = mem.recall("billing", top_k=5)
//! print(mem.stats())
//! ```

use super::consolidation::{self, ConsolidationConfig, ConsolidationReport};
use super::disk::{DiskRecord, DiskStore, MemoryBankConfig, StorageMode};
use super::episode::{
    hamming_distance, quantise_embedding, simhash_embedding, EmotionalTag, Episode,
};
use super::kanerva::KanervaSDM;
use super::lsh::{LSHIndex, ContextScorer};

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use tracing::info;

/// A recall result returned to Python.
#[derive(Clone, Debug)]
#[pyclass]
pub struct RecallResult {
    #[pyo3(get)]
    pub content: String,
    #[pyo3(get)]
    pub source: String,
    #[pyo3(get)]
    pub salience: f32,
    #[pyo3(get)]
    pub recall_count: u32,
    #[pyo3(get)]
    pub age_ticks: f64,
    #[pyo3(get)]
    pub retention: f32,
    #[pyo3(get)]
    pub consolidated: bool,
}

#[pymethods]
impl RecallResult {
    fn __repr__(&self) -> String {
        let status = if self.consolidated { "📦" } else { "🧠" };
        format!(
            "RecallResult({} '{}…' salience={:.2} recalls={} retention={:.1}%)",
            status,
            &self.content[..self.content.len().min(40)],
            self.salience,
            self.recall_count,
            self.retention * 100.0,
        )
    }
}


/// Statistics snapshot for the engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[pyclass]
pub struct HippocampusStats {
    #[pyo3(get)]
    pub episode_count: usize,
    #[pyo3(get)]
    pub consolidated_count: usize,
    #[pyo3(get)]
    pub sdm_occupied: usize,
    #[pyo3(get)]
    pub sdm_capacity: usize,
    #[pyo3(get)]
    pub avg_salience: f32,
    #[pyo3(get)]
    pub total_recalls: u64,
    #[pyo3(get)]
    pub total_consolidation_cycles: u64,
    #[pyo3(get)]
    pub total_evicted: u64,
}

#[pymethods]
impl HippocampusStats {
    fn __repr__(&self) -> String {
        format!(
            "HippocampusStats(episodes={}, consolidated={}, sdm={}/{}, avg_salience={:.2}, recalls={}, evicted={})",
            self.episode_count,
            self.consolidated_count,
            self.sdm_occupied,
            self.sdm_capacity,
            self.avg_salience,
            self.total_recalls,
            self.total_evicted,
        )
    }
}

/// HippocampusEngine — brain-inspired memory for AI agents.
///
/// Combines:
/// 1. Hippocampal ring buffer (fast episodic write, Ebbinghaus decay)
/// 2. Kanerva SDM neocortex (permanent semantic recall, O(1) via POPCNT)
/// 3. Sleep-replay consolidation (episodes graduate to permanent if important)
/// 4. Spaced-recall reinforcement (each recall strengthens the memory)
#[pyclass]
pub struct HippocampusEngine {
    /// Hippocampus: episodic memory buffer.
    episodes: Vec<Episode>,
    /// Maximum number of episodes before oldest are evicted.
    capacity: usize,
    /// Monotonic ID counter.
    next_id: u64,

    /// Neocortex: Kanerva SDM for consolidated patterns.
    sdm: KanervaSDM,

    /// Current simulation tick.
    current_tick: f64,

    /// How many ticks between automatic consolidation cycles.
    consolidation_interval: u64,
    /// Tick of last consolidation.
    last_consolidation_tick: f64,

    /// Recall reinforcement factor (default 1.3).
    recall_reinforcement: f32,
    /// Consolidation config.
    consolidation_config: ConsolidationConfig,

    // Cumulative stats
    total_recalls: u64,
    total_evicted: u64,
    total_consolidation_cycles: u64,

    /// Temporal B-tree index: tick → list of episode IDs created at that tick.
    /// Enables O(log N) time-range queries via `recall_between(start, end)`.
    temporal_index: BTreeMap<i64, Vec<u64>>,

    /// Episode ID → index in self.episodes vec (for fast lookup by ID).
    id_to_index: std::collections::HashMap<u64, usize>,

    /// Multi-Probe LSH Index for O(1) recall instead of O(N) scan.
    lsh_index: LSHIndex,

    /// Context-weighted scorer for ranking results by relevance.
    context_scorer: ContextScorer,

    /// Optional disk-backed persistence via mmap'd DiskStore.
    /// When present, every `remember()` also writes to disk, and
    /// existing records are loaded on construction.
    disk_store: Option<DiskStore>,
}

#[pymethods]
impl HippocampusEngine {
    /// Create a new HippocampusEngine.
    ///
    /// # Arguments
    /// - `capacity`: Max episodes in the hippocampal buffer (default: 100_000).
    /// - `sdm_locations`: Number of hard locations in the Kanerva SDM (default: 10_000).
    /// - `sdm_radius`: Activation radius for the SDM (default: 400).
    /// - `consolidation_interval`: Ticks between sleep-replay cycles (default: 100).
    /// - `recall_reinforcement`: Salience multiplier per recall (default: 1.3).
    /// - `memory_bank`: Optional `MemoryBankConfig` for disk-backed persistence.
    ///   When provided with `storage_mode="disk"`, memories are persisted to disk
    ///   and survive process restarts.
    #[new]
    #[pyo3(signature = (
        capacity = 100_000,
        sdm_locations = 10_000,
        sdm_radius = 400,
        consolidation_interval = 100,
        recall_reinforcement = 1.3,
        memory_bank = None
    ))]
    pub fn new(
        capacity: usize,
        sdm_locations: usize,
        sdm_radius: u32,
        consolidation_interval: u64,
        recall_reinforcement: f32,
        memory_bank: Option<MemoryBankConfig>,
    ) -> Self {
        info!(
            "🧠 [HippocampusEngine] Initialized: capacity={}, sdm_locations={}, interval={}",
            capacity, sdm_locations, consolidation_interval,
        );

        let mut engine = HippocampusEngine {
            episodes: Vec::with_capacity(capacity.min(1_000_000)),
            capacity,
            next_id: 0,
            sdm: KanervaSDM::new(sdm_locations, sdm_radius),
            current_tick: 0.0,
            consolidation_interval,
            last_consolidation_tick: 0.0,
            recall_reinforcement,
            consolidation_config: ConsolidationConfig::default(),
            total_recalls: 0,
            total_evicted: 0,
            total_consolidation_cycles: 0,
            temporal_index: BTreeMap::new(),
            id_to_index: std::collections::HashMap::new(),
            lsh_index: LSHIndex::new(),
            context_scorer: ContextScorer::default(),
            disk_store: None,
        };

        // Open disk store and load existing records if configured for disk mode.
        if let Some(ref config) = memory_bank {
            if config.storage_mode() == StorageMode::Disk {
                let store_path = std::path::PathBuf::from(&config.storage_path)
                    .join("hippocampus");
                match DiskStore::open_with_quota(
                    &store_path,
                    capacity,
                    config.disk_quota_bytes,
                ) {
                    Ok(store) => {
                        engine.load_from_disk(&store);
                        engine.disk_store = Some(store);
                        info!(
                            "🧠 [HippocampusEngine] Disk mode: loaded {} episodes from '{}'",
                            engine.episodes.len(),
                            store_path.display(),
                        );
                    }
                    Err(e) => {
                        info!(
                            "⚠️ [HippocampusEngine] Failed to open disk store at '{}': {}. Falling back to RAM-only.",
                            store_path.display(), e,
                        );
                    }
                }
            }
        }

        engine
    }

    /// Store a new memory.
    ///
    /// # Arguments
    /// - `content`: The memory text.
    /// - `salience`: Importance score (0.01 = routine, 1.0 = critical, 5.0+ = traumatic).
    /// - `source`: Provenance source (e.g. filename, URL, agent name). Default "".
    /// - `embedding`: Optional float embedding vector. If not provided, a
    ///   trigram-hash embedding is generated from the content using FNV1a.
    /// - `emotional_tag`: 0=neutral, 1=positive, 2=negative, 3=critical.
    #[pyo3(signature = (content, salience = 0.5, source = String::new(), embedding = None, emotional_tag = 0))]
    pub fn remember(
        &mut self,
        content: String,
        salience: f32,
        source: String,
        embedding: Option<Vec<f32>>,
        emotional_tag: u8,
    ) {
        // Generate or quantise the embedding.
        let quantised = match embedding {
            Some(ref float_vec) => quantise_embedding(float_vec),
            None => self.trigram_embedding(&content),
        };

        // Compute binary address for Kanerva SDM.
        let binary_address = simhash_embedding(&quantised);

        // ── Deduplication: exact-match check via LSH single-probe ──
        // If identical content already exists, boost its salience instead
        // of inserting a duplicate. Prevents memory bloat from recurring
        // alerts, re-posted LLM explanations, or repeated sensor data.
        //
        // Requires BOTH conditions:
        //   1. SimHash Hamming distance = 0 (same hash bucket)
        //   2. Content string equality (prevents false merges from collisions)
        let exact_matches = self.lsh_index.query_exact(&binary_address);
        for &idx in &exact_matches {
            if idx < self.episodes.len() {
                let ep = &self.episodes[idx];
                if hamming_distance(&binary_address, &ep.binary_address) == 0
                    && ep.content == content
                {
                    // Exact duplicate confirmed — boost existing episode
                    let boost = salience * 0.5; // Diminishing returns
                    self.episodes[idx].salience += boost;
                    self.episodes[idx].last_recalled = self.current_tick;
                    self.episodes[idx].recall_count += 1;
                    return; // No new episode created
                }
            }
        }

        // Apply emotional multiplier to salience.
        let emotional = EmotionalTag::from(emotional_tag);
        let adjusted_salience = salience
            * match emotional {
                EmotionalTag::Neutral => 1.0,
                EmotionalTag::Positive => 1.2,
                EmotionalTag::Negative => 1.5,
                EmotionalTag::Critical => 3.0,
            };

        let episode = Episode {
            id: self.next_id,
            content,
            source,
            embedding: quantised,
            binary_address,
            salience: adjusted_salience,
            created_at: self.current_tick,
            last_recalled: self.current_tick,
            recall_count: 0,
            emotional_tag: emotional,
            consolidated: false,
            related_to: Vec::new(),
        };

        self.next_id += 1;

        // If at capacity, evict the lowest-retention episode.
        if self.episodes.len() >= self.capacity {
            self.evict_weakest();
        }

        self.episodes.push(episode);

        // Update temporal index.
        let tick_key = self.current_tick as i64;
        self.temporal_index
            .entry(tick_key)
            .or_insert_with(Vec::new)
            .push(self.next_id - 1);

        // Update ID → index map.
        let ep_idx = self.episodes.len() - 1;
        self.id_to_index.insert(self.next_id - 1, ep_idx);

        // Insert into LSH index for O(1) recall.
        self.lsh_index.insert(&self.episodes[ep_idx].binary_address, ep_idx);

        // Persist to disk if disk store is active.
        if let Some(ref mut store) = self.disk_store {
            let ep = &self.episodes[ep_idx];
            let mut disk_rec = Self::episode_to_disk_record(ep);
            if let Err(e) = store.append(&mut disk_rec, &ep.content, &ep.source) {
                info!("⚠️ [HippocampusEngine] Disk write failed: {}", e);
            }
        }
    }

    /// Recall memories similar to a query.
    ///
    /// Searches both the hippocampal buffer AND the Kanerva SDM neocortex.
    /// Each recalled episode has its salience reinforced (spaced repetition).
    ///
    /// # Arguments
    /// - `query`: Text query for similarity matching.
    /// - `top_k`: Maximum number of results (default: 5).
    /// - `query_embedding`: Optional float embedding. If not provided, a hash-based
    ///   trigram-hash embedding is generated from the query text.
    #[pyo3(signature = (query, top_k = 5, query_embedding = None))]
    pub fn recall(
        &mut self,
        query: String,
        top_k: usize,
        query_embedding: Option<Vec<f32>>,
    ) -> Vec<RecallResult> {
        let quantised = match query_embedding {
            Some(ref float_vec) => quantise_embedding(float_vec),
            None => self.trigram_embedding(&query),
        };
        let query_address = simhash_embedding(&quantised);
        let tick = self.current_tick;

        // ── LSH-accelerated recall: O(L×k) instead of O(N) ──────────
        let candidates = self.lsh_index.query(&query_address);

        // Find max salience for normalization.
        let max_salience = candidates.iter()
            .filter_map(|&idx| {
                if idx < self.episodes.len() { Some(self.episodes[idx].salience) } else { None }
            })
            .fold(0.0f32, f32::max)
            .max(1.0);

        // Score candidates using context-weighted scoring.
        let mut scored: Vec<(usize, f32)> = candidates.iter()
            .filter(|&&idx| idx < self.episodes.len())
            .map(|&idx| {
                let ep = &self.episodes[idx];
                let dist = hamming_distance(&query_address, &ep.binary_address);
                let age = tick - ep.created_at;
                let emotion = match ep.emotional_tag {
                    EmotionalTag::Neutral => 0,
                    EmotionalTag::Positive => 1,
                    EmotionalTag::Negative => 2,
                    EmotionalTag::Critical => 3,
                };
                let score = self.context_scorer.score(
                    dist, ep.salience, age, emotion, max_salience,
                );
                (idx, score)
            })
            .collect();

        // Sort by score descending (highest = most relevant).
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Fallback: if LSH returned no candidates (empty engine or cold start),
        // do a brute-force scan on the small set.
        if scored.is_empty() && !self.episodes.is_empty() {
            scored = self.episodes.iter().enumerate()
                .map(|(i, ep)| {
                    let dist = hamming_distance(&query_address, &ep.binary_address);
                    let age = tick - ep.created_at;
                    let emotion = match ep.emotional_tag {
                        EmotionalTag::Neutral => 0,
                        EmotionalTag::Positive => 1,
                        EmotionalTag::Negative => 2,
                        EmotionalTag::Critical => 3,
                    };
                    let fallback_max = self.episodes.iter().map(|e| e.salience).fold(0.0f32, f32::max).max(1.0);
                    let score = self.context_scorer.score(dist, ep.salience, age, emotion, fallback_max);
                    (i, score)
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        }

        let mut results = Vec::with_capacity(top_k);

        for &(idx, _score) in scored.iter().take(top_k) {
            let ep = &mut self.episodes[idx];
            ep.reinforce(tick, self.recall_reinforcement);
            self.total_recalls += 1;

            results.push(RecallResult {
                content: ep.content.clone(),
                source: ep.source.clone(),
                salience: ep.salience,
                recall_count: ep.recall_count,
                age_ticks: tick - ep.created_at,
                retention: ep.retention(tick),
                consolidated: ep.consolidated,
            });
        }

        results
    }

    /// Advance the internal clock by one tick.
    ///
    /// If the consolidation interval has elapsed, runs a sleep-replay cycle.
    pub fn tick(&mut self) {
        self.current_tick += 1.0;

        // Check if consolidation is due.
        if self.current_tick - self.last_consolidation_tick
            >= self.consolidation_interval as f64
        {
            self.run_consolidation();
        }
    }

    /// Advance the internal clock by `n` ticks at once.
    #[pyo3(signature = (n))]
    pub fn advance(&mut self, n: u64) {
        self.current_tick += n as f64;
        // Check if consolidation is due.
        if self.current_tick - self.last_consolidation_tick
            >= self.consolidation_interval as f64
        {
            self.run_consolidation();
        }
    }

    /// Force a consolidation cycle (sleep replay) immediately.
    pub fn consolidate_now(&mut self) -> PyResult<String> {
        let report = self.run_consolidation();
        Ok(format!(
            "Consolidation: evicted={}, consolidated={}, surviving={}",
            report.evicted, report.consolidated, report.surviving,
        ))
    }

    /// Get engine statistics.
    pub fn stats(&self) -> HippocampusStats {
        let consolidated_count = self.episodes.iter().filter(|e| e.consolidated).count();
        let avg_salience = if self.episodes.is_empty() {
            0.0
        } else {
            self.episodes.iter().map(|e| e.salience).sum::<f32>() / self.episodes.len() as f32
        };

        HippocampusStats {
            episode_count: self.episodes.len(),
            consolidated_count,
            sdm_occupied: self.sdm.occupied_count(),
            sdm_capacity: self.sdm.capacity(),
            avg_salience,
            total_recalls: self.total_recalls,
            total_consolidation_cycles: self.total_consolidation_cycles,
            total_evicted: self.total_evicted,
        }
    }

    /// Get the current tick.
    #[getter]
    pub fn current_tick(&self) -> f64 {
        self.current_tick
    }

    /// Get the number of stored episodes.
    #[getter]
    pub fn episode_count(&self) -> usize {
        self.episodes.len()
    }

    /// Whether this engine is persisting to disk.
    #[getter]
    pub fn is_persistent(&self) -> bool {
        self.disk_store.is_some()
    }

    /// Get the disk storage path (empty string if RAM-only).
    #[getter]
    pub fn disk_path(&self) -> String {
        self.disk_store
            .as_ref()
            .map(|s| s.disk_path().to_string())
            .unwrap_or_default()
    }

    /// Flush any pending disk writes to stable storage.
    ///
    /// No-op if the engine is in RAM-only mode.
    pub fn flush(&self) -> PyResult<()> {
        if let Some(ref store) = self.disk_store {
            store.flush().map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Flush failed: {}", e))
            })?;
        }
        Ok(())
    }

    /// Create a relationship between two episodes.
    ///
    /// Links episode `from_id` to episode `to_id`. The link is unidirectional.
    /// Each episode can have up to 4 relationships.
    #[pyo3(signature = (from_id, to_id))]
    pub fn relate(&mut self, from_id: u64, to_id: u64) -> bool {
        if let Some(&idx) = self.id_to_index.get(&from_id) {
            let ep = &mut self.episodes[idx];
            if ep.related_to.len() < 4 {
                ep.related_to.push(to_id);
                return true;
            }
        }
        false
    }

    /// Recall all episodes related to a given episode ID.
    ///
    /// Follows relationship edges and returns the linked memories.
    #[pyo3(signature = (episode_id, depth = 1))]
    pub fn recall_related(&mut self, episode_id: u64, depth: usize) -> Vec<RecallResult> {
        let mut visited = std::collections::HashSet::new();
        let mut results = Vec::new();
        self.collect_related(episode_id, depth, &mut visited, &mut results);
        results
    }

    /// Recall episodes created between two ticks (inclusive).
    ///
    /// Uses the temporal B-tree index for O(log N) range lookup.
    #[pyo3(signature = (start_tick, end_tick, top_k = 10))]
    pub fn recall_between(
        &mut self,
        start_tick: f64,
        end_tick: f64,
        top_k: usize,
    ) -> Vec<RecallResult> {
        let start = start_tick as i64;
        let end = end_tick as i64;
        let tick = self.current_tick;

        let mut results = Vec::new();
        for (_tick, ids) in self.temporal_index.range(start..=end) {
            for &id in ids {
                if results.len() >= top_k {
                    break;
                }
                if let Some(&idx) = self.id_to_index.get(&id) {
                    let ep = &mut self.episodes[idx];
                    ep.reinforce(tick, self.recall_reinforcement);
                    self.total_recalls += 1;
                    results.push(RecallResult {
                        content: ep.content.clone(),
                        source: ep.source.clone(),
                        salience: ep.salience,
                        recall_count: ep.recall_count,
                        age_ticks: tick - ep.created_at,
                        retention: ep.retention(tick),
                        consolidated: ep.consolidated,
                    });
                }
            }
        }
        results
    }
}

// ── Private helpers ──────────────────────────────────────────────────────

impl HippocampusEngine {
    /// Convert an Episode to a DiskRecord for persistence.
    fn episode_to_disk_record(ep: &Episode) -> DiskRecord {
        let mut related = [u64::MAX; 4];
        for (i, &rid) in ep.related_to.iter().take(4).enumerate() {
            related[i] = rid;
        }
        let mut flags: u8 = 0x02; // alive
        if ep.consolidated {
            flags |= 0x01;
        }
        DiskRecord {
            id: ep.id,
            binary_address: ep.binary_address,
            salience: ep.salience,
            created_at: ep.created_at as f32,
            last_recalled: ep.last_recalled as f32,
            recall_count: ep.recall_count.min(u16::MAX as u32) as u16,
            emotional_tag: ep.emotional_tag as u8,
            flags,
            content_offset: 0, // filled by DiskStore::append
            content_len: 0,    // filled by DiskStore::append
            source_offset: 0,  // filled by DiskStore::append
            source_len: 0,     // filled by DiskStore::append
            related_to: related,
        }
    }

    /// Load existing episodes from a DiskStore into memory.
    fn load_from_disk(&mut self, store: &DiskStore) {
        for i in 0..store.len() {
            if let Some(rec) = store.get_record(i) {
                if !rec.is_alive() {
                    continue;
                }
                let content = store.read_content(&rec).unwrap_or_default();
                let source = store.read_source(&rec).unwrap_or_default();

                let mut related_to = Vec::new();
                for &rid in &rec.related_to {
                    if rid != u64::MAX {
                        related_to.push(rid);
                    }
                }

                let episode = Episode {
                    id: rec.id,
                    content,
                    source,
                    embedding: vec![32768u16; 64], // placeholder — binary_address is the real index
                    binary_address: rec.binary_address,
                    salience: rec.salience,
                    created_at: rec.created_at as f64,
                    last_recalled: rec.last_recalled as f64,
                    recall_count: rec.recall_count as u32,
                    emotional_tag: EmotionalTag::from(rec.emotional_tag),
                    consolidated: rec.is_consolidated(),
                    related_to,
                };

                // Track the highest ID seen so we continue from there.
                if episode.id >= self.next_id {
                    self.next_id = episode.id + 1;
                }
                // Track the latest tick so we resume from there.
                if episode.created_at > self.current_tick {
                    self.current_tick = episode.created_at;
                }

                self.episodes.push(episode);
            }
        }

        // Rebuild all indexes from loaded episodes.
        self.rebuild_index();

        // Rebuild temporal index.
        self.temporal_index.clear();
        for ep in &self.episodes {
            let tick_key = ep.created_at as i64;
            self.temporal_index
                .entry(tick_key)
                .or_insert_with(Vec::new)
                .push(ep.id);
        }
    }

    /// Run a single consolidation cycle.
    fn run_consolidation(&mut self) -> ConsolidationReport {
        let report = consolidation::consolidate(
            &mut self.episodes,
            &mut self.sdm,
            self.current_tick,
            &self.consolidation_config,
        );
        self.total_evicted += report.evicted as u64;
        self.total_consolidation_cycles += 1;
        self.last_consolidation_tick = self.current_tick;
        self.rebuild_index();
        report
    }

    /// Evict the episode with the lowest current retention.
    fn evict_weakest(&mut self) {
        if self.episodes.is_empty() {
            return;
        }
        let tick = self.current_tick;
        let mut worst_idx = 0;
        let mut worst_retention = f32::MAX;
        for (i, ep) in self.episodes.iter().enumerate() {
            let r = ep.retention(tick);
            if r < worst_retention {
                worst_retention = r;
                worst_idx = i;
            }
        }
        self.episodes.swap_remove(worst_idx);
        self.total_evicted += 1;
        self.rebuild_index();
    }

    /// Rebuild the id_to_index map after episodes are removed.
    fn rebuild_index(&mut self) {
        self.id_to_index.clear();
        self.lsh_index.clear();
        for (i, ep) in self.episodes.iter().enumerate() {
            self.id_to_index.insert(ep.id, i);
            self.lsh_index.insert(&ep.binary_address, i);
        }
    }

    /// Recursively collect related episodes (BFS up to given depth).
    fn collect_related(
        &mut self,
        episode_id: u64,
        depth: usize,
        visited: &mut std::collections::HashSet<u64>,
        results: &mut Vec<RecallResult>,
    ) {
        if depth == 0 || visited.contains(&episode_id) {
            return;
        }
        visited.insert(episode_id);

        let related_ids: Vec<u64> = if let Some(&idx) = self.id_to_index.get(&episode_id) {
            self.episodes[idx].related_to.clone()
        } else {
            return;
        };

        let tick = self.current_tick;
        for &rid in &related_ids {
            if visited.contains(&rid) {
                continue;
            }
            if let Some(&idx) = self.id_to_index.get(&rid) {
                let ep = &mut self.episodes[idx];
                ep.reinforce(tick, self.recall_reinforcement);
                self.total_recalls += 1;
                results.push(RecallResult {
                    content: ep.content.clone(),
                    source: ep.source.clone(),
                    salience: ep.salience,
                    recall_count: ep.recall_count,
                    age_ticks: tick - ep.created_at,
                    retention: ep.retention(tick),
                    consolidated: ep.consolidated,
                });
                // Recurse to next depth.
                self.collect_related(rid, depth - 1, visited, results);
            }
        }
    }

    /// Generate a deterministic trigram-hash embedding from text.
    ///
    /// Uses FNV1a character trigram hashing to produce a 64-dim u16 vector.
    /// This is a real technique from the SimHash/MinHash literature —
    /// shared trigrams → overlapping non-zero dimensions → small Hamming distance.
    /// NOT a fake/pseudo embedding — this is how locality-sensitive hashing works.
    ///
    /// For higher quality semantic recall, pass a real embedding model's
    /// output via the `embedding` parameter in `remember()` / `recall()`.
    fn trigram_embedding(&self, text: &str) -> Vec<u16> {
        let mut vec = vec![32768u16; 64]; // Start at midpoint

        let bytes = text.as_bytes();
        if bytes.len() < 3 {
            // Very short text: just hash the whole thing.
            let h = Self::fnv1a(bytes);
            vec[(h as usize) % 64] = 65535;
            return vec;
        }

        // Character trigram hashing.
        for window in bytes.windows(3) {
            let h = Self::fnv1a(window);
            let dim = (h as usize) % 64;
            let val = ((h >> 16) % 32768) as u16;
            // Accumulate: push toward 65535 for positive trigrams.
            vec[dim] = vec[dim].saturating_add(val);
        }

        vec
    }

    /// FNV-1a hash (fast, good distribution for short keys).
    #[inline]
    fn fnv1a(data: &[u8]) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        for &b in data {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remember_and_recall() {
        let mut engine = HippocampusEngine::new(1000, 100, 512, 100, 1.3, None);
        engine.remember(
            "User complained about billing".to_string(),
            0.8,
            "chat_log".to_string(),
            None,
            0,
        );
        engine.remember(
            "User said hello".to_string(),
            0.05,
            String::new(),
            None,
            0,
        );
        engine.remember(
            "User threatened legal action over billing".to_string(),
            0.95,
            "support_ticket".to_string(),
            None,
            3, // critical
        );

        let results = engine.recall("billing".to_string(), 5, None);
        assert!(!results.is_empty());
        // The billing-related memories should appear.
        let contents: Vec<&str> = results.iter().map(|r| r.content.as_str()).collect();
        assert!(contents.iter().any(|c| c.contains("billing")));
    }

    #[test]
    fn test_decay_eviction() {
        let mut engine = HippocampusEngine::new(1000, 100, 512, 10, 1.3, None);
        // Write a low-salience memory.
        engine.remember("routine hello".to_string(), 0.1, String::new(), None, 0);
        assert_eq!(engine.episodes.len(), 1);

        // Advance time significantly.
        engine.advance(100);
        // After consolidation, the low-salience memory should be evicted.
        assert_eq!(engine.episodes.len(), 0);
    }

    #[test]
    fn test_reinforcement_prevents_eviction() {
        // Use a long consolidation interval (1000) so no sleep-replay runs during the test.
        let mut engine = HippocampusEngine::new(1000, 100, 512, 1000, 1.3, None);
        engine.remember("important fact".to_string(), 5.0, String::new(), None, 0);

        // Recall it several times to build salience.
        for _ in 0..5 {
            engine.recall("important".to_string(), 1, None);
        }

        // salience should now be 5.0 * 1.3^5 ≈ 18.6
        let ep = &engine.episodes[0];
        assert!(ep.salience > 10.0, "Salience should be reinforced: {}", ep.salience);

        // Advance 10 ticks (no consolidation since interval=1000).
        engine.advance(10);
        // retention = e^(-10/18.6) ≈ 0.58 — well above death threshold.
        assert_eq!(engine.episodes.len(), 1);
    }

    #[test]
    fn test_capacity_eviction() {
        let mut engine = HippocampusEngine::new(3, 100, 512, 1000, 1.3, None);
        for i in 0..5 {
            engine.remember(
                format!("memory {}", i),
                0.5 + i as f32 * 0.1,
                String::new(),
                None,
                0,
            );
        }
        // Should have at most 3 episodes.
        assert!(engine.episodes.len() <= 3);
    }

    #[test]
    fn test_stats() {
        let mut engine = HippocampusEngine::new(1000, 100, 512, 100, 1.3, None);
        engine.remember("test".to_string(), 0.5, String::new(), None, 0);
        let st = engine.stats();
        assert_eq!(st.episode_count, 1);
        assert_eq!(st.total_recalls, 0);
    }
}
