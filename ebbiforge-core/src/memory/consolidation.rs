//! Sleep-Replay Consolidation
//!
//! Implements the core CLS (Complementary Learning Systems) consolidation
//! algorithm: replay hippocampal episodes, identify which have earned
//! permanent status, and migrate them to the neocortex (Kanerva SDM).

use super::episode::Episode;
use super::kanerva::KanervaSDM;

/// Thresholds governing the consolidation process.
#[derive(Clone, Debug)]
pub struct ConsolidationConfig {
    /// Minimum retention level below which an episode is evicted (forgotten).
    /// Default: 0.01 (1% retention).
    pub death_threshold: f32,
    /// Minimum retention level AND recall count to consolidate to neocortex.
    /// Default: 0.5 (50% retention) + at least 2 recalls.
    pub consolidation_retention_threshold: f32,
    /// Minimum recall count for an episode to be considered for consolidation.
    pub min_recall_count: u32,
    /// Salience reduction factor applied after consolidation.
    /// The hippocampal copy is weakened since the neocortex now owns it.
    pub post_consolidation_salience_factor: f32,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        ConsolidationConfig {
            death_threshold: 0.01,
            consolidation_retention_threshold: 0.5,
            min_recall_count: 2,
            post_consolidation_salience_factor: 0.5,
        }
    }
}

/// Result of a single consolidation cycle.
#[derive(Clone, Debug, Default)]
pub struct ConsolidationReport {
    /// Number of episodes that decayed below death threshold and were evicted.
    pub evicted: usize,
    /// Number of episodes consolidated to the neocortex.
    pub consolidated: usize,
    /// Number of episodes that survived in the hippocampus.
    pub surviving: usize,
    /// Total episodes processed.
    pub total_processed: usize,
}

/// Run one consolidation cycle (sleep replay).
///
/// This function:
/// 1. Scans all hippocampal episodes
/// 2. Evicts those whose retention has dropped below `death_threshold`
/// 3. Consolidates high-retention, frequently-recalled episodes to the Kanerva SDM
/// 4. Returns a report of what happened
///
/// The caller retains the `surviving` episodes in the hippocampus.
pub fn consolidate(
    episodes: &mut Vec<Episode>,
    sdm: &mut KanervaSDM,
    current_tick: f64,
    config: &ConsolidationConfig,
) -> ConsolidationReport {
    let total = episodes.len();
    let mut report = ConsolidationReport {
        total_processed: total,
        ..Default::default()
    };

    let mut surviving = Vec::with_capacity(total);

    for mut ep in episodes.drain(..) {
        let retention = ep.retention(current_tick);

        if retention < config.death_threshold {
            // Memory has decayed beyond recovery → evict.
            report.evicted += 1;
            continue;
        }

        if retention >= config.consolidation_retention_threshold
            && ep.recall_count >= config.min_recall_count
            && !ep.consolidated
        {
            // This memory is important AND repeatedly accessed.
            // Consolidate to neocortex (Kanerva SDM).
            let content: Vec<f32> = ep
                .embedding
                .iter()
                .map(|&v| (v as f32 / 32768.0) - 1.0) // De-quantise to [-1, 1]
                .collect();
            sdm.write(&ep.binary_address, &content);
            ep.consolidated = true;
            // Weaken the hippocampal copy — neocortex owns it now.
            ep.salience *= config.post_consolidation_salience_factor;
            report.consolidated += 1;
        }

        surviving.push(ep);
        report.surviving += 1;
    }

    *episodes = surviving;
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::episode::{EmotionalTag, simhash_embedding};

    fn make_episode(id: u64, salience: f32, created_at: f64, recall_count: u32) -> Episode {
        let embedding = vec![32768u16; 64];
        let binary_address = simhash_embedding(&embedding);
        Episode {
            id,
            content: format!("memory_{}", id),
            source: String::new(),
            embedding,
            binary_address,
            salience,
            created_at,
            last_recalled: created_at,
            recall_count,
            emotional_tag: EmotionalTag::Neutral,
            consolidated: false,
            related_to: Vec::new(),
        }
    }

    #[test]
    fn test_eviction() {
        let mut sdm = KanervaSDM::new(100, 512);
        let config = ConsolidationConfig::default();
        // Episode created at tick 0 with salience 0.1 → at tick 100, retention ≈ 0.
        let mut episodes = vec![make_episode(1, 0.1, 0.0, 0)];
        let report = consolidate(&mut episodes, &mut sdm, 100.0, &config);
        assert_eq!(report.evicted, 1);
        assert_eq!(report.surviving, 0);
        assert!(episodes.is_empty());
    }

    #[test]
    fn test_consolidation() {
        let mut sdm = KanervaSDM::new(100, 512);
        let config = ConsolidationConfig::default();
        // High salience, 3 recalls, created recently → should consolidate.
        let mut episodes = vec![make_episode(1, 100.0, 0.0, 3)];
        let report = consolidate(&mut episodes, &mut sdm, 1.0, &config);
        assert_eq!(report.consolidated, 1);
        assert_eq!(report.surviving, 1);
        assert!(episodes[0].consolidated);
        // Salience should be reduced.
        assert!((episodes[0].salience - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_surviving_without_consolidation() {
        let mut sdm = KanervaSDM::new(100, 512);
        let config = ConsolidationConfig::default();
        // Moderate salience, 0 recalls → survives but not consolidated.
        let mut episodes = vec![make_episode(1, 10.0, 0.0, 0)];
        let report = consolidate(&mut episodes, &mut sdm, 1.0, &config);
        assert_eq!(report.consolidated, 0);
        assert_eq!(report.surviving, 1);
        assert!(!episodes[0].consolidated);
    }
}
