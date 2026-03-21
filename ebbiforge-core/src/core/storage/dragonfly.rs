//! DragonflyDB/Redis storage backend
//!
//! Uses the redis crate which is compatible with DragonflyDB.
//! DragonflyDB is 25x faster than Redis with multi-threaded architecture.

use async_trait::async_trait;
use pyo3::prelude::*;
use redis::aio::ConnectionManager;
use redis::AsyncCommands;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

use super::{KeyValueStore, StorageError, StorageResult};

/// DragonflyDB client for session state and agent memory
pub struct DragonflyClient {
    connection: Arc<RwLock<Option<ConnectionManager>>>,
    url: String,
}

impl DragonflyClient {
    pub fn new(url: &str) -> Self {
        DragonflyClient {
            connection: Arc::new(RwLock::new(None)),
            url: url.to_string(),
        }
    }

    async fn get_connection(&self) -> StorageResult<ConnectionManager> {
        {
            let conn = self.connection.read().await;
            if let Some(ref c) = *conn {
                return Ok(c.clone());
            }
        }

        // Need to establish connection
        let client = redis::Client::open(self.url.as_str())
            .map_err(|e| StorageError::ConnectionFailed(e.to_string()))?;

        let manager = ConnectionManager::new(client)
            .await
            .map_err(|e| StorageError::ConnectionFailed(e.to_string()))?;

        {
            let mut conn = self.connection.write().await;
            *conn = Some(manager.clone());
        }

        info!("🐉 [DragonflyDB] Connected to {}", self.url);
        Ok(manager)
    }
}

#[async_trait]
impl KeyValueStore for DragonflyClient {
    async fn save(&self, key: &str, value: &str) -> StorageResult<()> {
        let mut conn = self.get_connection().await?;
        conn.set::<_, _, ()>(key, value)
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;
        Ok(())
    }

    async fn load(&self, key: &str) -> StorageResult<String> {
        let mut conn = self.get_connection().await?;
        let value: Option<String> = conn
            .get(key)
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;

        value.ok_or_else(|| StorageError::NotFound(key.to_string()))
    }

    async fn delete(&self, key: &str) -> StorageResult<()> {
        let mut conn = self.get_connection().await?;
        conn.del::<_, ()>(key)
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;
        Ok(())
    }

    async fn exists(&self, key: &str) -> StorageResult<bool> {
        let mut conn = self.get_connection().await?;
        let exists: bool = conn
            .exists(key)
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;
        Ok(exists)
    }

    async fn save_with_ttl(&self, key: &str, value: &str, ttl_secs: u64) -> StorageResult<()> {
        let mut conn = self.get_connection().await?;
        conn.set_ex::<_, _, ()>(key, value, ttl_secs)
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;
        Ok(())
    }
}

/// Python wrapper for DragonflyDB client
#[pyclass]
pub struct DragonflyStore {
    client: Arc<DragonflyClient>,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl DragonflyStore {
    #[new]
    pub fn new(url: String) -> PyResult<Self> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(DragonflyStore {
            client: Arc::new(DragonflyClient::new(&url)),
            runtime: Arc::new(runtime),
        })
    }

    /// Save a value
    pub fn save(&self, key: String, value: String) -> PyResult<()> {
        let client = self.client.clone();
        self.runtime
            .block_on(async move { client.save(&key, &value).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Load a value
    pub fn load(&self, key: String) -> PyResult<String> {
        let client = self.client.clone();
        self.runtime
            .block_on(async move { client.load(&key).await })
            .map_err(|e| pyo3::exceptions::PyKeyError::new_err(e.to_string()))
    }

    /// Delete a key
    pub fn delete(&self, key: String) -> PyResult<()> {
        let client = self.client.clone();
        self.runtime
            .block_on(async move { client.delete(&key).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Check if key exists
    pub fn exists(&self, key: String) -> PyResult<bool> {
        let client = self.client.clone();
        self.runtime
            .block_on(async move { client.exists(&key).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Save with TTL (seconds)
    pub fn save_with_ttl(&self, key: String, value: String, ttl_secs: u64) -> PyResult<()> {
        let client = self.client.clone();
        self.runtime
            .block_on(async move { client.save_with_ttl(&key, &value, ttl_secs).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}
