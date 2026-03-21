use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Key, Nonce,
};
use hex;
use pyo3::prelude::*;
use rand::RngCore;

#[pyclass]
pub struct SecureVault {
    cipher: Aes256Gcm,
}

#[pymethods]
impl SecureVault {
    #[new]
    #[pyo3(signature = (key_hex = None))]
    pub fn new(key_hex: Option<String>) -> PyResult<Self> {
        let key_bytes = match key_hex {
            Some(hex_str) => hex::decode(hex_str).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Invalid hex key: {}", e))
            })?,
            None => {
                let mut key = [0u8; 32];
                rand::thread_rng().fill_bytes(&mut key);
                key.to_vec()
            }
        };

        if key_bytes.len() != 32 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Key must be 32 bytes (256 bits)",
            ));
        }

        let key = Key::<Aes256Gcm>::from_slice(&key_bytes);
        let cipher = Aes256Gcm::new(key);

        Ok(SecureVault { cipher })
    }

    /// Generate a fresh 32-byte key as hex
    #[staticmethod]
    pub fn generate_key() -> String {
        let mut key = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut key);
        hex::encode(key)
    }

    /// Encrypt plaintext using AES-256-GCM
    /// Returns "nonce_hex:ciphertext_hex"
    pub fn encrypt(&self, plaintext: &str) -> PyResult<String> {
        let mut nonce_bytes = [0u8; 12];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = self
            .cipher
            .encrypt(nonce, plaintext.as_bytes())
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Encryption failed: {}", e))
            })?;

        Ok(format!(
            "{}:{}",
            hex::encode(nonce_bytes),
            hex::encode(ciphertext)
        ))
    }

    /// Decrypt ciphertext using AES-256-GCM
    pub fn decrypt(&self, encrypted_data: &str) -> PyResult<String> {
        let parts: Vec<&str> = encrypted_data.split(':').collect();
        if parts.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid encrypted format. Expected 'nonce:ciphertext'",
            ));
        }

        let nonce_bytes = hex::decode(parts[0]).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid nonce hex: {}", e))
        })?;
        let ciphertext = hex::decode(parts[1]).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid ciphertext hex: {}", e))
        })?;

        let nonce = Nonce::from_slice(&nonce_bytes);
        let plaintext_bytes = self
            .cipher
            .decrypt(nonce, ciphertext.as_ref())
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Decryption failed: {}", e))
            })?;

        String::from_utf8(plaintext_bytes).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid UTF-8 plaintext: {}", e))
        })
    }
}
