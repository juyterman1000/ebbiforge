//! Storage backends for CogOps persistence
//!
//! Provides pluggable storage for:
//! - DragonflyDB (session cache, Redis-compatible)
//! - Vector Database (vector embeddings)

pub mod dragonfly;
pub mod remote_vector;

use async_trait::async_trait;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Result type for storage operations
pub type StorageResult<T> = Result<T, StorageError>;

/// Storage error types
#[derive(Debug, Clone)]
pub enum StorageError {
    ConnectionFailed(String),
    NotFound(String),
    SerializationError(String),
    OperationFailed(String),
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageError::ConnectionFailed(msg) => write!(f, "Connection failed: {}", msg),
            StorageError::NotFound(msg) => write!(f, "Not found: {}", msg),
            StorageError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            StorageError::OperationFailed(msg) => write!(f, "Operation failed: {}", msg),
        }
    }
}

impl std::error::Error for StorageError {}

/// Trait for key-value storage backends (DragonflyDB, Redis)
#[async_trait]
pub trait KeyValueStore: Send + Sync {
    /// Save a buffer as JSON
    async fn save(&self, key: &str, value: &str) -> StorageResult<()>;

    /// Load a buffer by key
    async fn load(&self, key: &str) -> StorageResult<String>;

    /// Delete a key
    async fn delete(&self, key: &str) -> StorageResult<()>;

    /// Check if key exists
    async fn exists(&self, key: &str) -> StorageResult<bool>;

    /// Set with TTL (seconds)
    async fn save_with_ttl(&self, key: &str, value: &str, ttl_secs: u64) -> StorageResult<()>;
}

/// Trait for vector storage backends
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Insert or update a vector
    async fn upsert(
        &self,
        id: &str,
        vector: Vec<f32>,
        payload: Option<serde_json::Value>,
    ) -> StorageResult<()>;

    /// Search for similar vectors
    async fn search(&self, vector: Vec<f32>, limit: usize) -> StorageResult<Vec<SearchResult>>;

    /// Delete a vector by ID
    async fn delete(&self, id: &str) -> StorageResult<()>;
}

/// Search result from vector store
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct SearchResult {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub score: f32,
    #[pyo3(get)]
    pub payload: Option<String>,
}

#[pymethods]
impl SearchResult {
    #[new]
    pub fn new(id: String, score: f32, payload: Option<String>) -> Self {
        SearchResult { id, score, payload }
    }
}

/// Storage configuration for Python
#[pyclass]
#[derive(Clone)]
pub struct StorageConfig {
    #[pyo3(get, set)]
    pub dragonfly_url: String,
    #[pyo3(get, set)]
    pub vector_db_url: String,
    #[pyo3(get, set)]
    pub collection_name: String,
    #[pyo3(get, set)]
    pub vector_size: usize,
}

#[pymethods]
impl StorageConfig {
    #[new]
    #[pyo3(signature = (dragonfly_url = "redis://localhost:6379".to_string(), vector_db_url = "http://localhost:6333".to_string(), collection_name = "cogops_memory".to_string(), vector_size = 1536))]
    pub fn new(
        dragonfly_url: String,
        vector_db_url: String,
        collection_name: String,
        vector_size: usize,
    ) -> Self {
        StorageConfig {
            dragonfly_url,
            vector_db_url,
            collection_name,
            vector_size,
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "StorageConfig(dragonfly='{}', vector_db='{}', collection='{}')",
            self.dragonfly_url, self.vector_db_url, self.collection_name
        )
    }
}
