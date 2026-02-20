//! PII detection and replacement engine.
//!
//! This module provides the core functionality for detecting and replacing
//! personally identifiable information (PII) in text.
//!
//! # Example
//!
//! ```
//! use nym::engine::{Detector, DetectorConfig, Replacer, ReplacerConfig};
//!
//! // Create a detector with default settings
//! let detector = Detector::with_defaults();
//!
//! // Detect PII in text
//! let text = "Contact me at john@example.com or 555-123-4567";
//! let matches = detector.detect(text);
//!
//! // Replace detected PII
//! let mut replacer = Replacer::with_defaults();
//! let (anonymized, replacements) = replacer.replace_all(text, &matches);
//!
//! println!("Anonymized: {}", anonymized);
//! ```

pub mod detector;
pub mod formats;
pub mod ner;
pub mod patterns;
pub mod replacer;

pub use detector::{Detector, DetectorConfig, PiiMatch};
pub use formats::{JsonPiiMatch, detect_json, process_json};
#[expect(unused_imports, reason = "Conditionally used with ner feature")]
pub use ner::set_exit_code;
pub use patterns::{BUILTIN_PATTERNS, Confidence, PiiCategory, PiiPattern, get_pattern};
pub use replacer::{Replacement, ReplacementStrategy, Replacer, ReplacerConfig};
