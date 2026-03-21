use crate::core::security::{sign_data, verify_signature, AgentIdentity, TrustStore};
use crate::TrajectoryPoint;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::info;

/// Failure context for cross-pollination
#[derive(Clone, Debug, Serialize, Deserialize)]
#[pyclass]
pub struct FailureContext {
    #[pyo3(get, set)]
    pub trace_id: String,
    #[pyo3(get, set)]
    pub input: String,
    #[pyo3(get, set)]
    pub output: String,
    #[pyo3(get, set)]
    pub error: Option<String>,
    #[pyo3(get, set)]
    pub tags: Vec<String>,
    pub trajectory: Vec<TrajectoryPoint>,
}

#[pymethods]
impl FailureContext {
    #[new]
    pub fn new(trace_id: String, input: String, output: String) -> Self {
        FailureContext {
            trace_id,
            input,
            output,
            error: None,
            tags: Vec::new(),
            trajectory: Vec::new(),
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

/// Experience Pack for sharing learnings across instances
#[derive(Clone, Debug, Serialize, Deserialize)]
#[pyclass]
pub struct ExperiencePack {
    #[pyo3(get, set)]
    pub version: String,
    #[pyo3(get, set)]
    pub source: String,
    #[pyo3(get, set)]
    pub timestamp: u64,
    pub lessons: Vec<FailureContext>,
    #[pyo3(get, set)]
    pub signature: Option<Vec<u8>>,
    #[pyo3(get, set)]
    pub sender_pubkey: Option<String>,
}

#[pymethods]
impl ExperiencePack {
    #[new]
    pub fn new(source: String) -> Self {
        ExperiencePack {
            version: "4.0.0".to_string(),
            source,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            lessons: Vec::new(),
            signature: None,
            sender_pubkey: None,
        }
    }

    pub fn add_lesson(&mut self, lesson: FailureContext) {
        self.lessons.push(lesson);
    }

    pub fn lesson_count(&self) -> usize {
        self.lessons.len()
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

/// Cross-Pollination engine for sharing learnings
#[pyclass]
pub struct CrossPollination {
    identity: Option<AgentIdentity>,
    trust_store: Option<Py<TrustStore>>,
}

#[pymethods]
impl CrossPollination {
    #[new]
    #[pyo3(signature = (identity = None, trust_store = None))]
    pub fn new(identity: Option<AgentIdentity>, trust_store: Option<Py<TrustStore>>) -> Self {
        CrossPollination {
            identity,
            trust_store,
        }
    }

    /// Export experience from a source
    pub fn export_experience(&self, source_id: String, lessons_json: String) -> ExperiencePack {
        info!("Exporting lessons from {}...", source_id);

        let lessons_res: Result<Vec<FailureContext>, _> = serde_json::from_str(&lessons_json);
        let lessons = lessons_res.unwrap_or_default();

        // Anonymize data
        let anonymized: Vec<FailureContext> = lessons
            .into_iter()
            .map(|mut l| {
                l.input = "[ANONYMIZED]".to_string();
                l
            })
            .collect();

        let mut pack = ExperiencePack::new(source_id);
        for lesson in anonymized {
            pack.add_lesson(lesson);
        }

        // Sign the pack if identity is present
        if let Some(ref id) = self.identity {
            pack.sender_pubkey = Some(id.pubkey.clone());

            // Sign the serialized content (excluding signature field)
            let serialized = pack.to_json();
            let sig = sign_data(id, serialized.as_bytes());
            pack.signature = Some(sig);
            info!("🔒 Signed experience pack with pubkey: {}", id.pubkey);
        }

        pack
    }

    /// Ingest experience pack into local memory (via safety shield)
    pub fn pollinate(&self, py: Python, pack_json: String) -> PyResult<Vec<String>> {
        let pack: ExperiencePack = serde_json::from_str(&pack_json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid pack JSON: {}", e))
        })?;

        // Verify signature if trust store is present
        if let Some(ref py_store) = self.trust_store {
            let store = py_store.borrow(py);

            let pubkey = pack.sender_pubkey.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("Missing sender pubkey in signed pack")
            })?;

            let sig = pack.signature.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("Missing signature in signed pack")
            })?;

            // Reconstruct pack without signature to verify
            let mut check_pack = pack.clone();
            check_pack.signature = None;
            let serialized = check_pack.to_json();

            if !verify_signature(pubkey, serialized.as_bytes(), sig) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Invalid experience pack signature",
                ));
            }

            // Check if pubkey is in trust store
            if !store.is_trusted(&pack.source, pubkey) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Sender '{}' is not in the trust store",
                    pack.source
                )));
            }

            info!("Verified signature from trusted agent: {}", pack.source);
        }

        info!(
            "🌸 Ingesting {} lessons from {}...",
            pack.lesson_count(),
            pack.source
        );

        // Return trajectory JSONs for loading into safety shield
        Ok(pack
            .lessons
            .iter()
            .map(|l| serde_json::to_string(&l.trajectory).unwrap_or_default())
            .collect())
    }
}
