//! Security and Trust layer for Ebbiforge
//!
//! Provides cryptographic identity and signing for agents to prevent
//! manipulation and poisoning attacks.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use parking_lot::RwLock;
use pyo3::prelude::*;
use rand::rngs::OsRng;
use rand::RngCore;
use std::collections::HashMap;

/// Cryptographic identity for an AI agent
#[pyclass]
#[derive(Clone)]
pub struct AgentIdentity {
    pub signing_key_bytes: [u8; 32],
    #[pyo3(get)]
    pub pubkey: String,
}

#[pymethods]
impl AgentIdentity {
    /// Generate a new random identity
    #[staticmethod]
    pub fn generate() -> Self {
        let mut key_bytes = [0u8; 32];
        OsRng.fill_bytes(&mut key_bytes);

        let signing_key = SigningKey::from_bytes(&key_bytes);
        let verifying_key = VerifyingKey::from(&signing_key);

        let pubkey_hex = hex::encode(verifying_key.to_bytes());

        AgentIdentity {
            signing_key_bytes: key_bytes,
            pubkey: pubkey_hex,
        }
    }

    /// Create from existing private key hex
    #[staticmethod]
    pub fn from_hex(hex_key: String) -> PyResult<Self> {
        let bytes = hex::decode(hex_key)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        if bytes.len() != 32 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid key length",
            ));
        }

        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&bytes);

        let signing_key = SigningKey::from_bytes(&key_bytes);
        let verifying_key = VerifyingKey::from(&signing_key);
        let pubkey_hex = hex::encode(verifying_key.to_bytes());

        Ok(AgentIdentity {
            signing_key_bytes: key_bytes,
            pubkey: pubkey_hex,
        })
    }

    pub fn __repr__(&self) -> String {
        format!("AgentIdentity(pubkey='{}')", self.pubkey)
    }
}

impl AgentIdentity {
    pub fn get_keys(&self) -> (SigningKey, VerifyingKey) {
        let s_key = SigningKey::from_bytes(&self.signing_key_bytes);
        let v_key = VerifyingKey::from(&s_key);
        (s_key, v_key)
    }
}

/// A store of trusted public keys
#[pyclass]
pub struct TrustStore {
    trusted_keys: RwLock<HashMap<String, [u8; 32]>>,
}

#[pymethods]
impl TrustStore {
    #[new]
    pub fn new() -> Self {
        TrustStore {
            trusted_keys: RwLock::new(HashMap::new()),
        }
    }

    /// Add a trusted agent and its public key
    pub fn add_trusted_agent(&self, agent_id: String, pubkey_hex: String) -> PyResult<()> {
        let bytes = hex::decode(pubkey_hex)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        if bytes.len() != 32 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid public key length",
            ));
        }

        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&bytes);

        let mut keys = self.trusted_keys.write();
        keys.insert(agent_id, key_bytes);
        Ok(())
    }

    /// Check if an agent is trusted
    pub fn is_trusted(&self, agent_id: &str, pubkey_hex: &str) -> bool {
        let bytes_res = hex::decode(pubkey_hex);
        if let Ok(bytes) = bytes_res {
            if bytes.len() == 32 {
                let mut key_bytes = [0u8; 32];
                key_bytes.copy_from_slice(&bytes);

                let keys = self.trusted_keys.read();
                if let Some(stored_key) = keys.get(agent_id) {
                    return stored_key == &key_bytes;
                }
            }
        }
        false
    }

    pub fn remove_trusted_agent(&self, agent_id: &str) {
        let mut keys = self.trusted_keys.write();
        keys.remove(agent_id);
    }
}

impl Default for TrustStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Sign data using an identity
pub fn sign_data(identity: &AgentIdentity, data: &[u8]) -> Vec<u8> {
    let (signing_key, _) = identity.get_keys();
    let signature: Signature = signing_key.sign(data);
    signature.to_bytes().to_vec()
}

/// Verify data using a public key
pub fn verify_signature(pubkey_hex: &str, data: &[u8], signature_bytes: &[u8]) -> bool {
    let b_res = hex::decode(pubkey_hex);
    if let Ok(bytes) = b_res {
        if bytes.len() == 32 {
            let mut key_bytes = [0u8; 32];
            key_bytes.copy_from_slice(&bytes);

            if let Ok(verifying_key) = VerifyingKey::from_bytes(&key_bytes) {
                if signature_bytes.len() == 64 {
                    let mut sig_bytes = [0u8; 64];
                    sig_bytes.copy_from_slice(signature_bytes);
                    if let Ok(signature) = Signature::from_slice(&sig_bytes) {
                        return verifying_key.verify(data, &signature).is_ok();
                    }
                }
            }
        }
    }
    false
}
