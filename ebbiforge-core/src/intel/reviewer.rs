use crate::core::middleware::{EbbiforgeContext, Middleware};
use pyo3::prelude::*;
use regex::Regex;
use tracing::info;

/// Code Quality Guard - Reviews code for secrets, TODOs, and console spam.
#[pyclass]
pub struct CodeQualityGuard {}

#[pymethods]
impl CodeQualityGuard {
    #[new]
    pub fn new() -> Self {
        CodeQualityGuard {}
    }

    /// Review code content for issues
    pub fn review_code(&self, content: String, filename: String) -> Vec<String> {
        let mut issues: Vec<String> = Vec::new();

        // 1. Check for secrets
        if let Ok(re) = Regex::new(r"sk-[a-zA-Z0-9]{20,}") {
            if re.is_match(&content) {
                issues.push(
                    "Hardcoded API Secret detected! Use environment variables instead.".to_string(),
                );
            }
        }

        if let Ok(re) = Regex::new(r#"API_KEY\s*=\s*['"][^'"]+['"]"#) {
            if re.is_match(&content) {
                issues.push(
                    "Hardcoded API_KEY detected! Use environment variables instead.".to_string(),
                );
            }
        }

        // 2. Check for TODOs
        if content.contains("TODO") {
            issues.push(
                "Found 'TODO' comments. Please finish implementation before committing."
                    .to_string(),
            );
        }

        // 3. Check for console spam
        let log_count = content.matches("console.log").count()
            + content.matches("info!(").count() // Replaced println! with tracing info!
            + content.matches("warn!(").count() // Added tracing warn!
            + content.matches("error!(").count() // Added tracing error!
            + content.matches("debug!(").count() // Added tracing debug!
            + content.matches("trace!(").count() // Added tracing trace!
            + content.matches("print(").count();
        if log_count > 5 {
            issues.push(format!(
                "Too many log statements ({}). Clean up debug code.",
                log_count
            ));
        }

        // 4. Check for unsafe blocks in Rust
        if filename.ends_with(".rs") && content.contains("unsafe") {
            issues
                .push("Found 'unsafe' block. Ensure this is necessary and documented.".to_string());
        }

        issues
    }
}

impl Middleware for CodeQualityGuard {
    fn name(&self) -> &str {
        "CodeQualityGuard"
    }

    fn before_step(&self, _ctx: &mut EbbiforgeContext) -> Result<(), String> {
        info!("🧐 [CodeReview] Ready to review code quality.");
        Ok(())
    }
}
