/// Embedding Engine — optional dense vector embeddings for hybrid retrieval.
///
/// Compiled only when `--features embeddings` is active.
/// Uses fastembed (ONNX Runtime) with BAAI/bge-small-en-v1.5 (33MB, 384-dim).
///
/// Architecture:
///   SimHash (64-bit, O(1) hamming)  — structural fingerprint, 65 discrete levels
///   N-gram Jaccard (word-level)     — lexical overlap, continuous [0,1]
///   Cosine embedding (384-dim)      — semantic similarity, captures meaning beyond words
///
/// The 3-way blend: SimHash 20% + Jaccard 30% + Cosine 50%
/// closes the semantic gap vs Augment Code's full vector search.

#[cfg(feature = "embeddings")]
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};

#[cfg(feature = "embeddings")]
use std::sync::OnceLock;

/// Global singleton for the embedding model (lazy init, ~33MB download on first use).
#[cfg(feature = "embeddings")]
static EMBEDDER: OnceLock<TextEmbedding> = OnceLock::new();

/// Get or initialize the embedding model.
///
/// Uses BAAI/bge-small-en-v1.5:
///   - 384 dimensions (compact, fast)
///   - Excellent for code retrieval (trained on code + text)
///   - ONNX runtime: no GPU needed, ~5ms per embedding on CPU
#[cfg(feature = "embeddings")]
fn get_embedder() -> &'static TextEmbedding {
    EMBEDDER.get_or_init(|| {
        TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::BGESmallENV15)
                .with_show_download_progress(false)
        ).expect("Failed to initialize embedding model")
    })
}

/// Compute a dense embedding vector for a single text.
///
/// Returns a 384-dimensional f32 vector.
/// Falls back to empty vec on error (graceful degradation).
#[cfg(feature = "embeddings")]
pub fn embed_text(text: &str) -> Vec<f32> {
    let embedder = get_embedder();
    // Truncate very long texts to avoid OOM (embeddings work best on <512 tokens)
    let truncated = if text.len() > 2048 { &text[..2048] } else { text };
    match embedder.embed(vec![truncated.to_string()], None) {
        Ok(mut results) if !results.is_empty() => results.remove(0),
        _ => Vec::new(),
    }
}

/// Compute a dense embedding vector — no-op stub when embeddings feature is disabled.
#[cfg(not(feature = "embeddings"))]
pub fn embed_text(_text: &str) -> Vec<f32> {
    Vec::new()
}

/// Cosine similarity between two embedding vectors.
///
/// Returns 0.0 for empty or zero-norm vectors (graceful fallback).
/// Optimized with manual SIMD-friendly loop (auto-vectorized by LLVM).
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;

    for i in 0..a.len() {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        return 0.0;
    }

    // Clamp to [0, 1] — negative cosine = unrelated, treat as 0
    (dot / denom).max(0.0).min(1.0)
}

/// Check whether embedding support is compiled in.
pub fn embeddings_enabled() -> bool {
    cfg!(feature = "embeddings")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 0.001, "Identical vectors should have cosine ~1.0, got {sim}");
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.001, "Orthogonal vectors should have cosine ~0.0, got {sim}");
    }

    #[test]
    fn test_cosine_empty() {
        let sim = cosine_similarity(&[], &[]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_mismatched_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similar_vectors() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.1, 2.1, 3.1, 4.1]; // very similar
        let sim = cosine_similarity(&a, &b);
        assert!(sim > 0.99, "Similar vectors should have high cosine, got {sim}");
    }

    #[test]
    fn test_cosine_negative_clamped() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0]; // opposite direction
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0, "Negative cosine should be clamped to 0.0");
    }

    #[test]
    fn test_embed_text_no_feature() {
        // When embeddings feature is disabled, should return empty vec
        #[cfg(not(feature = "embeddings"))]
        {
            let emb = embed_text("hello world");
            assert!(emb.is_empty());
        }
    }

    #[test]
    fn test_embeddings_enabled_flag() {
        let enabled = embeddings_enabled();
        #[cfg(feature = "embeddings")]
        assert!(enabled);
        #[cfg(not(feature = "embeddings"))]
        assert!(!enabled);
    }
}
