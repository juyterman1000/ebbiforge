//! Rate Limiter for enterprise cost control
//!
//! Prevents runaway API costs by limiting requests per agent/time window.

use parking_lot::RwLock;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// Rate limit configuration
#[derive(Clone)]
#[pyclass]
pub struct RateLimitConfig {
    #[pyo3(get, set)]
    pub requests_per_minute: u32,
    #[pyo3(get, set)]
    pub tokens_per_minute: u32,
    #[pyo3(get, set)]
    pub actions_per_hour: u32,
}

#[pymethods]
impl RateLimitConfig {
    #[new]
    #[pyo3(signature = (requests_per_minute = 60, tokens_per_minute = 100000, actions_per_hour = 1000))]
    pub fn new(requests_per_minute: u32, tokens_per_minute: u32, actions_per_hour: u32) -> Self {
        RateLimitConfig {
            requests_per_minute,
            tokens_per_minute,
            actions_per_hour,
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self::new(60, 100000, 1000)
    }
}

/// Sliding window rate limiter
struct SlidingWindow {
    window_start: Instant,
    window_duration: Duration,
    count: u32,
    limit: u32,
}

impl SlidingWindow {
    fn new(limit: u32, window_secs: u64) -> Self {
        SlidingWindow {
            window_start: Instant::now(),
            window_duration: Duration::from_secs(window_secs),
            count: 0,
            limit,
        }
    }

    fn check_and_increment(&mut self) -> bool {
        let now = Instant::now();

        // Reset window if expired
        if now.duration_since(self.window_start) >= self.window_duration {
            self.window_start = now;
            self.count = 0;
        }

        if self.count >= self.limit {
            false
        } else {
            self.count += 1;
            true
        }
    }

    fn remaining(&self) -> u32 {
        if self.count >= self.limit {
            0
        } else {
            self.limit - self.count
        }
    }
}

/// Rate limiter result
#[derive(Clone)]
#[pyclass]
pub struct RateLimitResult {
    #[pyo3(get)]
    pub allowed: bool,
    #[pyo3(get)]
    pub remaining: u32,
    #[pyo3(get)]
    pub reason: String,
}

#[pymethods]
impl RateLimitResult {
    pub fn __repr__(&self) -> String {
        if self.allowed {
            format!("RateLimitResult(ALLOWED, remaining={})", self.remaining)
        } else {
            format!("RateLimitResult(BLOCKED: {})", self.reason)
        }
    }
}

/// Enterprise rate limiter
#[pyclass]
pub struct RateLimiter {
    config: RateLimitConfig,
    request_windows: RwLock<HashMap<String, SlidingWindow>>,
    action_windows: RwLock<HashMap<String, SlidingWindow>>,
}

#[pymethods]
impl RateLimiter {
    #[new]
    #[pyo3(signature = (config = None))]
    pub fn new(config: Option<RateLimitConfig>) -> Self {
        let cfg = config.unwrap_or_default();
        info!(
            "⏱️  [RateLimiter] Initialized: {}/min requests, {}/hr actions",
            cfg.requests_per_minute, cfg.actions_per_hour
        );
        RateLimiter {
            config: cfg,
            request_windows: RwLock::new(HashMap::new()),
            action_windows: RwLock::new(HashMap::new()),
        }
    }

    /// Check if a request is allowed
    pub fn check_request(&self, agent_id: String) -> RateLimitResult {
        let mut windows = self.request_windows.write();

        let window = windows
            .entry(agent_id.clone())
            .or_insert_with(|| SlidingWindow::new(self.config.requests_per_minute, 60));

        if window.check_and_increment() {
            RateLimitResult {
                allowed: true,
                remaining: window.remaining(),
                reason: String::new(),
            }
        } else {
            warn!(
                "🚫 [RateLimiter] Request blocked for {}: rate limit exceeded",
                agent_id
            );
            RateLimitResult {
                allowed: false,
                remaining: 0,
                reason: format!(
                    "Rate limit exceeded: {} requests/minute",
                    self.config.requests_per_minute
                ),
            }
        }
    }

    /// Check if an action is allowed
    pub fn check_action(&self, agent_id: String) -> RateLimitResult {
        let mut windows = self.action_windows.write();

        let window = windows
            .entry(agent_id.clone())
            .or_insert_with(|| SlidingWindow::new(self.config.actions_per_hour, 3600));

        if window.check_and_increment() {
            RateLimitResult {
                allowed: true,
                remaining: window.remaining(),
                reason: String::new(),
            }
        } else {
            warn!(
                "🚫 [RateLimiter] Action blocked for {}: hourly limit exceeded",
                agent_id
            );
            RateLimitResult {
                allowed: false,
                remaining: 0,
                reason: format!(
                    "Action limit exceeded: {} actions/hour",
                    self.config.actions_per_hour
                ),
            }
        }
    }

    /// Get current usage stats
    pub fn get_stats(&self, agent_id: String) -> String {
        let req_windows = self.request_windows.read();
        let act_windows = self.action_windows.read();

        let req_remaining = req_windows
            .get(&agent_id)
            .map(|w| w.remaining())
            .unwrap_or(self.config.requests_per_minute);
        let act_remaining = act_windows
            .get(&agent_id)
            .map(|w| w.remaining())
            .unwrap_or(self.config.actions_per_hour);

        format!(
            "RateLimitStats(requests_remaining={}, actions_remaining={})",
            req_remaining, act_remaining
        )
    }
}
