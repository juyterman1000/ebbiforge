//! PRISM-Inspired Anisotropic Spectral Optimizer
//!
//! Implements quasi-second-order RL weight tuning using running covariance
//! and Eigenvalue Decomposition, adapted from "PRISM: Structured Optimization
//! via Anisotropic Spectral Shaping".
//!
//! Instead of isotropic (scalar) learning rates, this tracks the 4x4 covariance
//! matrix of the feature gradients (Recency, Frequency, Semantic, Entropy).
//! It computes the eigendecomposition $C = Q \Lambda Q^T$ and applies
//! anisotropic damping in high-variance (noisy) sub-spaces:
//! $w_{t+1} = w_t + \alpha Q \Lambda^{-1/2} Q^T g$.
//!
//! Because our state space is exactly 4D, we perform exact eigendecomposition
//! (Jacobi method) rather than the approximate polar decomposition needed for
//! 100M+ parameter neural networks.

use std::f64;
use serde::{Serialize, Deserialize};

/// A 4x4 symmetric matrix for tracking gradient covariance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SymMatrix4 {
    pub data: [[f64; 4]; 4],
}

impl SymMatrix4 {
    pub fn new() -> Self {
        SymMatrix4 { data: [[0.0; 4]; 4] }
    }

    pub fn identity() -> Self {
        let mut m = Self::new();
        for i in 0..4 { m.data[i][i] = 1.0; }
        m
    }

    /// Update running covariance: $C = \beta C + (1-\beta) g g^T$
    pub fn update_ema(&mut self, g: &[f64; 4], beta: f64) {
        for i in 0..4 {
            for j in 0..4 {
                self.data[i][j] = beta * self.data[i][j] + (1.0 - beta) * (g[i] * g[j]);
            }
        }
    }

    /// Computes Eigenvalue Decomposition $C = Q \Lambda Q^T$ using the Cyclic Jacobi method.
    /// Returns (Q, Eigenvalues), where Q columns are eigenvectors.
    pub fn jacobi_eigendecomposition(&self) -> (SymMatrix4, [f64; 4]) {
        let mut a = self.clone();
        let mut q = Self::identity();
        let mut iters = 0;
        let max_iters = 50;
        let eps = 1e-9;

        while iters < max_iters {
            // Find max off-diagonal element
            let mut max_val = 0.0;
            let mut p = 0;
            let mut r = 1;
            for i in 0..3 {
                for j in (i + 1)..4 {
                    let val = a.data[i][j].abs();
                    if val > max_val {
                        max_val = val;
                        p = i;
                        r = j;
                    }
                }
            }

            if max_val < eps {
                break; // Converged
            }

            // Compute Jacobi rotation
            let app = a.data[p][p];
            let arr = a.data[r][r];
            let apr = a.data[p][r];
            let theta = 0.5 * (2.0 * apr / (app - arr + 1e-15)).atan();
            let c = theta.cos();
            let s = theta.sin();

            // Apply rotation A' = J^T A J
            for i in 0..4 {
                if i != p && i != r {
                    let aip = a.data[i][p];
                    let air = a.data[i][r];
                    a.data[i][p] = c * aip - s * air;
                    a.data[p][i] = a.data[i][p];
                    a.data[i][r] = s * aip + c * air;
                    a.data[r][i] = a.data[i][r];
                }
            }
            a.data[p][p] = c * c * app - 2.0 * s * c * apr + s * s * arr;
            a.data[r][r] = s * s * app + 2.0 * s * c * apr + c * c * arr;
            a.data[p][r] = 0.0;
            a.data[r][p] = 0.0;

            // Apply rotation Q' = Q J
            for i in 0..4 {
                let qip = q.data[i][p];
                let qir = q.data[i][r];
                q.data[i][p] = c * qip - s * qir;
                q.data[i][r] = s * qip + c * qir;
            }

            iters += 1;
        }

        let eigenvalues = [a.data[0][0], a.data[1][1], a.data[2][2], a.data[3][3]];
        (q, eigenvalues)
    }
}

/// Anisotropic Spectral Optimizer (PRISM-lite for 4D Context Weights)
///
/// Value Residual: Like ResFormer's x0 skip connections, this optimizer mixes
/// PRISM-learned weights with the initial default weights via a learned
/// residual_lambda:
///   final_weights = residual_lambda * prism_weights + (1 - residual_lambda) * initial_weights
/// This prevents weight drift during early adaptation when feedback is sparse.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrismOptimizer {
    pub covariance: SymMatrix4,
    pub beta: f64,
    pub learning_rate: f64,
    pub epsilon: f64,
    /// Initial weights snapshot (the "x0" in ResFormer terminology).
    /// Weights always mix back toward these defaults via residual_lambda.
    #[serde(default = "default_initial_weights")]
    pub initial_weights: [f64; 4],
    /// Residual mixing coefficient [0, 1]. 0 = pure initial, 1 = pure PRISM.
    /// Starts at 0.1 (mostly initial) and grows toward 1.0 as confidence builds.
    #[serde(default = "default_residual_lambda")]
    pub residual_lambda: f64,
    /// Number of updates received — used to grow residual_lambda over time.
    #[serde(default)]
    pub update_count: u64,
}

fn default_initial_weights() -> [f64; 4] { [0.30, 0.25, 0.25, 0.20] }
fn default_residual_lambda() -> f64 { 0.1 }

impl PrismOptimizer {
    pub fn new(learning_rate: f64) -> Self {
        let mut cov = SymMatrix4::identity();
        // Initialize with small epsilon identity to prevent division by zero gracefully
        for i in 0..4 { cov.data[i][i] = 1e-4; }
        PrismOptimizer {
            covariance: cov,
            beta: 0.95, // exponential moving average decay
            learning_rate,
            epsilon: 1e-6,
            initial_weights: [0.30, 0.25, 0.25, 0.20],
            residual_lambda: 0.1,
            update_count: 0,
        }
    }

    /// Applies Anisotropic Spectral Gain to a gradient vector.
    /// Computes $P g = Q \Lambda^{-1/2} Q^T g$ and returns the update $\Delta w = \alpha P g$.
    pub fn compute_update(&mut self, g: &[f64; 4]) -> [f64; 4] {
        self.update_count += 1;

        // Grow residual_lambda toward 1.0: sigmoid warmup over ~100 updates.
        // At 0 updates: lambda = 0.1 (mostly initial weights)
        // At 50 updates: lambda ~ 0.5
        // At 100+ updates: lambda ~ 0.9 (mostly PRISM weights)
        self.residual_lambda = 1.0 / (1.0 + (-0.05 * (self.update_count as f64 - 50.0)).exp());

        // 1. Update running covariance
        self.covariance.update_ema(g, self.beta);

        // 2. Eigendecomposition: C = Q \Lambda Q^T
        let (q, eigenvalues) = self.covariance.jacobi_eigendecomposition();

        // 3. Spectral Shaping (Inverse Square Root: \Lambda^{-1/2})
        // This dampens directions with high variance (noise) and boosts clean signals.
        let mut lambda_inv_sqrt = [0.0; 4];
        for i in 0..4 {
            lambda_inv_sqrt[i] = 1.0 / (eigenvalues[i].abs() + self.epsilon).sqrt();
        }

        // 4. Compute Q \Lambda^{-1/2} Q^T g
        // First, project gradient into eigenspace: v = Q^T g
        let mut v = [0.0; 4];
        for i in 0..4 {
            for j in 0..4 {
                v[i] += q.data[j][i] * g[j];
            }
        }

        // Apply spectral shaping: v' = \Lambda^{-1/2} v
        for i in 0..4 {
            v[i] *= lambda_inv_sqrt[i];
        }

        // Project back to feature space: update = \alpha Q v'
        let mut step = [0.0; 4];
        for i in 0..4 {
            for j in 0..4 {
                step[i] += q.data[i][j] * v[j];
            }
            step[i] *= self.learning_rate;
        }

        step
    }

    /// Apply value residual mixing: blend current weights with initial defaults.
    ///
    /// Based on the ResFormer residual pattern (x0 skip connection):
    /// prevents PRISM from drifting too far from sensible defaults during
    /// early adaptation when feedback data is sparse.
    ///
    /// Returns mixed weights: lambda * current + (1-lambda) * initial
    pub fn apply_residual(&self, current_weights: &[f64; 4]) -> [f64; 4] {
        let lam = self.residual_lambda;
        let mut mixed = [0.0; 4];
        for i in 0..4 {
            mixed[i] = lam * current_weights[i] + (1.0 - lam) * self.initial_weights[i];
        }
        mixed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jacobi_eigendecomposition_diagonal() {
        let mut mat = SymMatrix4::new();
        mat.data[0][0] = 5.0;
        mat.data[1][1] = 3.0;
        mat.data[2][2] = 2.0;
        mat.data[3][3] = 1.0;

        let (q, eigs) = mat.jacobi_eigendecomposition();
        
        // Eigenvalues should match diagonal
        assert!((eigs[0] - 5.0).abs() < 1e-6 || (eigs[1] - 5.0).abs() < 1e-6 || (eigs[2] - 5.0).abs() < 1e-6 || (eigs[3] - 5.0).abs() < 1e-6);
        
        // Q should be identity (possibly permuted/signed)
        let det_q = q.data[0][0]*q.data[1][1]*q.data[2][2]*q.data[3][3];
        assert!(det_q.abs() > 0.9);
    }

    #[test]
    fn test_anisotropic_shaping_dampens_noise() {
        let mut optim = PrismOptimizer::new(0.1);
        
        // Flood the optimizer with high variance (noise) in dimension 0
        for _ in 0..100 {
            optim.compute_update(&[10.0, 0.1, 0.1, 0.1]);
            optim.compute_update(&[-10.0, 0.1, 0.1, 0.1]);
        }
        
        // Now, a single unit gradient in all directions
        let step = optim.compute_update(&[1.0, 1.0, 1.0, 1.0]);
        
        // Dimension 0 should be heavily damped compared to 1, 2, 3
        assert!(step[0].abs() < step[1].abs() * 0.1, "PRISM failed to anisotropically damp the noisy dimension");
    }

    #[test]
    fn test_value_residual_prevents_drift() {
        let mut optim = PrismOptimizer::new(0.1);
        optim.initial_weights = [0.30, 0.25, 0.25, 0.20];

        // Early on (few updates), residual_lambda should be low,
        // so apply_residual should pull weights back toward initial.
        let drifted = [0.80, 0.05, 0.10, 0.05];
        let mixed_early = optim.apply_residual(&drifted);

        // Lambda starts at ~0.1, so mixed should be ~90% initial + 10% drifted
        // mixed[0] ~ 0.1*0.80 + 0.9*0.30 = 0.35
        assert!(mixed_early[0] < 0.50, "Early residual should anchor to initial weights, got {}", mixed_early[0]);

        // After many updates, lambda approaches 1.0 and residual fades out
        for _ in 0..200 {
            optim.compute_update(&[0.1, 0.1, 0.1, 0.1]);
        }
        let mixed_late = optim.apply_residual(&drifted);
        // Lambda should be ~0.9+, so mixed should be mostly drifted
        assert!(mixed_late[0] > 0.60, "Late residual should trust PRISM weights, got {}", mixed_late[0]);
    }

    #[test]
    fn test_residual_lambda_sigmoid_warmup() {
        let mut optim = PrismOptimizer::new(0.1);
        // At 0 updates, lambda should be small
        assert!(optim.residual_lambda < 0.2);

        // After 50 updates, lambda should be ~0.5
        for _ in 0..50 {
            optim.compute_update(&[0.1, 0.1, 0.1, 0.1]);
        }
        assert!((optim.residual_lambda - 0.5).abs() < 0.15,
            "Lambda at 50 updates should be near 0.5, got {}", optim.residual_lambda);

        // After 200 updates, lambda should be close to 1.0
        for _ in 0..150 {
            optim.compute_update(&[0.1, 0.1, 0.1, 0.1]);
        }
        assert!(optim.residual_lambda > 0.85,
            "Lambda at 200 updates should be > 0.85, got {}", optim.residual_lambda);
    }
}
