use parking_lot::RwLock;
use pyo3::prelude::*;
use std::sync::Arc;
use tracing::info;

/// A Zero-Copy memory store for sharing embeddings between agents.
/// Follows memory-safe views over raw bytes principles.
#[pyclass]
pub struct SharedMemoryStore {
    buffer: Arc<RwLock<Vec<f32>>>,
    vector_size: usize,
    capacity: usize,
}

#[pymethods]
impl SharedMemoryStore {
    #[new]
    pub fn new(capacity: usize, vector_size: usize) -> Self {
        let total_size = capacity * vector_size;
        info!(
            "[SharedMemory] Allocated {} floats (Capacity: {} vectors of size {}).",
            total_size, capacity, vector_size
        );

        SharedMemoryStore {
            buffer: Arc::new(RwLock::new(vec![0.0; total_size])),
            vector_size,
            capacity,
        }
    }

    /// Default constructor for 100 vectors of 1536 dimensions (Standard embedding size)
    #[staticmethod]
    pub fn default() -> Self {
        Self::new(100, 1536)
    }

    /// Write a vector to the store at the given index.
    /// O(N) copy where N = vector_size.
    pub fn write(&self, index: usize, vector: Vec<f32>) -> PyResult<()> {
        if index >= self.capacity {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "Index out of bounds",
            ));
        }
        if vector.len() != self.vector_size {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Vector size mismatch: expected {}, got {}",
                self.vector_size,
                vector.len()
            )));
        }

        let mut data = self.buffer.write();
        let start = index * self.vector_size;
        for (i, val) in vector.into_iter().enumerate() {
            data[start + i] = val;
        }

        Ok(())
    }

    /// Read a vector from the store at the given index.
    /// Returns a copy (for Python safety). Internal Rust code can use zero-copy reads.
    pub fn read(&self, index: usize) -> PyResult<Vec<f32>> {
        if index >= self.capacity {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "Index out of bounds",
            ));
        }

        let data = self.buffer.read();
        let start = index * self.vector_size;
        let end = start + self.vector_size;

        Ok(data[start..end].to_vec())
    }

    /// Get the total capacity (number of vectors)
    pub fn get_capacity(&self) -> usize {
        self.capacity
    }

    /// Get the vector size (dimensions per vector)
    pub fn get_vector_size(&self) -> usize {
        self.vector_size
    }

    /// Fork the store (creates a new reference to the same underlying buffer).
    /// This is TRUE zero-copy: both stores share the same memory.
    pub fn fork(&self) -> Self {
        SharedMemoryStore {
            buffer: self.buffer.clone(),
            vector_size: self.vector_size,
            capacity: self.capacity,
        }
    }

    /// Compute cosine similarity between two vectors at given indices.
    pub fn cosine_similarity(&self, index_a: usize, index_b: usize) -> PyResult<f64> {
        if index_a >= self.capacity || index_b >= self.capacity {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "Index out of bounds",
            ));
        }

        let data = self.buffer.read();
        let start_a = index_a * self.vector_size;
        let start_b = index_b * self.vector_size;

        let mut dot = 0.0_f64;
        let mut mag_a = 0.0_f64;
        let mut mag_b = 0.0_f64;

        for i in 0..self.vector_size {
            let a = data[start_a + i] as f64;
            let b = data[start_b + i] as f64;
            dot += a * b;
            mag_a += a * a;
            mag_b += b * b;
        }

        let denom = mag_a.sqrt() * mag_b.sqrt();
        if denom == 0.0 {
            return Ok(0.0);
        }

        Ok(dot / denom)
    }
}
