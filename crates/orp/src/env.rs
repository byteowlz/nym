//! Process-global ONNX Runtime environment management.
//!
//! `ort` 2.0-rc.15 allows exactly one environment per process and offers no
//! public accessor for the existing one, so every consumer must share a single
//! create-once handle. The first caller creates it; later callers receive a
//! clone. A host application that owns its own environment should create it
//! through this module *before* any `Model` is constructed, so all sessions in
//! the process share one environment.

use ort::environment::Environment;
use std::sync::OnceLock;

static ENVIRONMENT: OnceLock<Result<Environment, String>> = OnceLock::new();

/// The process-global environment, creating it on first use.
///
/// # Errors
/// Returns an error when the ONNX Runtime environment cannot be created.
pub fn global_environment() -> ort::Result<Environment> {
    ENVIRONMENT
        .get_or_init(|| ort::environment::init().build().map_err(|e| e.to_string()))
        .clone()
        .map_err(ort::Error::new)
}
