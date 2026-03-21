//! Remote vector database backend
//!
//! This module provides a client for remote vector databases with sub-millisecond latency.
//! Perfect for storing embeddings and similarity search for AI agents.

use async_trait::async_trait;
use pyo3::prelude::*;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, DeletePointsBuilder, Distance, PointId, PointStruct,
    SearchPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::Qdrant;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

use super::{SearchResult, StorageError, StorageResult, VectorStore};

/// Professional client for vector storage
pub struct VectorDbClient {
    client: Arc<RwLock<Option<Qdrant>>>,
    url: String,
    collection: String,
    vector_size: u64,
}

impl VectorDbClient {
    pub fn new(url: &str, collection: &str, vector_size: usize) -> Self {
        VectorDbClient {
            client: Arc::new(RwLock::new(None)),
            url: url.to_string(),
            collection: collection.to_string(),
            vector_size: vector_size as u64,
        }
    }

    async fn get_client(&self) -> StorageResult<Qdrant> {
        {
            let c = self.client.read().await;
            if let Some(ref client) = *c {
                return Ok(client.clone());
            }
        }

        // Create new client connecting to the remote vector database
        let client = Qdrant::from_url(&self.url)
            .build()
            .map_err(|e| StorageError::ConnectionFailed(e.to_string()))?;

        // Ensure collection exists
        let collections = client
            .list_collections()
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;

        let exists = collections
            .collections
            .iter()
            .any(|c| c.name == self.collection);

        if !exists {
            client
                .create_collection(
                    CreateCollectionBuilder::new(self.collection.clone()).vectors_config(
                        VectorParamsBuilder::new(self.vector_size, Distance::Cosine),
                    ),
                )
                .await
                .map_err(|e| StorageError::OperationFailed(e.to_string()))?;

            info!(
                "🔷 [VectorDB] Created collection '{}' (size: {})",
                self.collection, self.vector_size
            );
        }

        {
            let mut c = self.client.write().await;
            *c = Some(client.clone());
        }

        info!("🔷 [VectorDB] Connected to {}", self.url);
        Ok(client)
    }
}

#[async_trait]
impl VectorStore for VectorDbClient {
    async fn upsert(
        &self,
        id: &str,
        vector: Vec<f32>,
        payload: Option<serde_json::Value>,
    ) -> StorageResult<()> {
        let client = self.get_client().await?;

        let mut payload_map: HashMap<String, qdrant_client::qdrant::Value> = HashMap::new();

        if let Some(p) = payload {
            if let serde_json::Value::Object(map) = p {
                for (k, v) in map {
                    if let serde_json::Value::String(s) = v {
                        payload_map.insert(k, s.into());
                    }
                }
            }
        }

        let point = PointStruct::new(id.to_string(), vector, payload_map);

        client
            .upsert_points(UpsertPointsBuilder::new(
                self.collection.clone(),
                vec![point],
            ))
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;

        Ok(())
    }

    async fn search(&self, vector: Vec<f32>, limit: usize) -> StorageResult<Vec<SearchResult>> {
        let client = self.get_client().await?;

        let results = client
            .search_points(
                SearchPointsBuilder::new(self.collection.clone(), vector, limit as u64)
                    .with_payload(true),
            )
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;

        let search_results = results
            .result
            .into_iter()
            .map(|r| {
                let id = match r.id {
                    Some(point_id) => format!("{:?}", point_id),
                    None => "unknown".to_string(),
                };
                SearchResult {
                    id,
                    score: r.score,
                    payload: Some(serde_json::to_string(&r.payload).unwrap_or_default()),
                }
            })
            .collect();

        Ok(search_results)
    }

    async fn delete(&self, id: &str) -> StorageResult<()> {
        let client = self.get_client().await?;

        let point_id: PointId = PointId::from(id.to_string());
        client
            .delete_points(DeletePointsBuilder::new(self.collection.clone()).points(vec![point_id]))
            .await
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;

        Ok(())
    }
}

/// Python wrapper for the vector client
#[pyclass]
pub struct RemoteVectorStore {
    client: Arc<VectorDbClient>,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl RemoteVectorStore {
    #[new]
    #[pyo3(signature = (url, collection = "cogops_memory".to_string(), vector_size = 1536))]
    pub fn new(url: String, collection: String, vector_size: usize) -> PyResult<Self> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(RemoteVectorStore {
            client: Arc::new(VectorDbClient::new(&url, &collection, vector_size)),
            runtime: Arc::new(runtime),
        })
    }

    /// Insert or update a vector
    pub fn upsert(&self, id: String, vector: Vec<f32>, payload: Option<String>) -> PyResult<()> {
        let client = self.client.clone();
        let payload_json = payload.and_then(|p| serde_json::from_str(&p).ok());

        self.runtime
            .block_on(async move { client.upsert(&id, vector, payload_json).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Search for similar vectors
    pub fn search(&self, vector: Vec<f32>, limit: usize) -> PyResult<Vec<SearchResult>> {
        let client = self.client.clone();
        self.runtime
            .block_on(async move { client.search(vector, limit).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Delete a vector by ID
    pub fn delete(&self, id: String) -> PyResult<()> {
        let client = self.client.clone();
        self.runtime
            .block_on(async move { client.delete(&id).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}
