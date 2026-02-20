//! Named Entity Recognition using GLiNER models.
//!
//! This module provides NER-based PII detection using the GLiNER model family.
//! It complements regex-based detection by identifying entities that are hard
//! to match with patterns, such as person names and free-form addresses.
//!
//! # ONNX Runtime macOS Crash Workaround
//!
//! Due to a known bug in ONNX Runtime 1.21+ on macOS, the ONNX environment cleanup
//! crashes with "mutex lock failed: Invalid argument" during process exit.
//! See: <https://github.com/microsoft/onnxruntime/issues/24579>
//!
//! This module implements a workaround by:
//! 1. Leaking the ONNX Runtime environment Arc to prevent cleanup
//! 2. Using ManuallyDrop for the NerDetector to prevent model cleanup
//!
//! This is a memory leak, but it only happens once per process and the memory
//! is reclaimed by the OS when the process exits anyway.

#[cfg(feature = "ner")]
use gliner::model::GLiNER;
#[cfg(feature = "ner")]
use gliner::model::input::text::TextInput;
#[cfg(feature = "ner")]
use gliner::model::params::Parameters;
#[cfg(feature = "ner")]
use gliner::model::pipeline::span::SpanMode;
#[cfg(feature = "ner")]
use orp::params::RuntimeParameters;
#[cfg(feature = "ner")]
use std::path::Path;

#[cfg(feature = "ner")]
use crate::engine::detector::PiiMatch;
#[cfg(feature = "ner")]
use crate::engine::patterns::Confidence;
#[cfg(feature = "ner")]
use crate::engine::patterns::PiiCategory;

/// Labels for PII entity detection.
/// These are passed to GLiNER for zero-shot entity extraction.
///
/// Note: GLiNER is a zero-shot NER model, so we can add any labels we want.
/// The model will try to extract entities matching these semantic concepts.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "Public API - used by consumers")
)]
pub const PII_LABELS: &[&str] = &[
    // Name components - try to get first/last name separately
    "person",
    "first_name",
    "last_name",
    // Organizations
    "organization",
    // Location types
    "street_address",
    "city",
    "state",
    "country",
    // Contact
    "phone_number",
    // Temporal
    "date",
    "date_of_birth",
    "time",
];

/// Maps GLiNER entity labels to nym pattern names.
#[cfg(feature = "ner")]
fn label_to_pattern_name(label: &str) -> &'static str {
    match label.to_lowercase().as_str() {
        // Name-related
        "person" => "person",
        "first_name" | "firstname" | "given_name" | "givenname" => "first_name",
        "last_name" | "lastname" | "surname" | "family_name" => "last_name",
        // Organizations
        "organization" | "company" | "org" => "organization",
        // Location types
        "street_address" | "address" => "street_address",
        "city" => "city",
        "state" | "province" | "region" => "state",
        "country" => "country",
        "location" => "location",
        // Contact
        "phone_number" | "phone" => "phone_ner",
        // Temporal
        "date" => "date",
        "date_of_birth" | "dob" | "birthday" | "birthdate" => "date_of_birth",
        "time" => "time",
        _ => "ner_entity",
    }
}

/// Maps GLiNER entity labels to PII categories.
#[cfg(feature = "ner")]
fn label_to_category(label: &str) -> PiiCategory {
    match label.to_lowercase().as_str() {
        // Names are identity
        "person" | "first_name" | "firstname" | "last_name" | "lastname" => PiiCategory::Identity,
        // Organizations
        "organization" | "company" | "org" => PiiCategory::Other,
        // Locations are contact info
        "street_address" | "address" | "city" | "state" | "country" | "location" => {
            PiiCategory::Contact
        }
        // Phone is contact
        "phone_number" | "phone" => PiiCategory::Contact,
        // Dates can be identity (birthdate) or other
        "date_of_birth" | "dob" | "birthday" | "birthdate" => PiiCategory::Identity,
        "date" | "time" => PiiCategory::Other,
        _ => PiiCategory::Other,
    }
}

/// Default chunk size in words (GLiNER has 384 token limit, using conservative word count)
#[cfg(feature = "ner")]
const DEFAULT_CHUNK_SIZE: usize = 250;

/// Default overlap in words to avoid splitting entities at boundaries
#[cfg(feature = "ner")]
const DEFAULT_CHUNK_OVERLAP: usize = 50;

/// Minimum text length in characters for NER processing.
/// GLiNER fails with reshape errors on very short inputs.
#[cfg(feature = "ner")]
const MIN_TEXT_LENGTH: usize = 3;

/// Minimum word count for NER processing.
/// Single words or empty strings cause ONNX runtime errors.
#[cfg(feature = "ner")]
const MIN_WORD_COUNT: usize = 2;

/// Global exit code for macOS crash workaround.
/// Set by the program before exit to preserve the intended exit status.
#[cfg(all(feature = "ner", target_os = "macos"))]
static EXIT_CODE: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

/// Set the exit code for macOS ONNX Runtime crash workaround.
/// Call this before exiting to preserve the exit status.
///
/// This is a no-op on non-macOS platforms or when NER is not enabled.
#[cfg(all(feature = "ner", target_os = "macos"))]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "Public API - used by consumers")
)]
pub fn set_exit_code(code: i32) {
    EXIT_CODE.store(code, std::sync::atomic::Ordering::SeqCst);
}

/// Set the exit code (no-op stub for non-macOS).
#[cfg(not(all(feature = "ner", target_os = "macos")))]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "Public API - used by consumers")
)]
pub fn set_exit_code(_code: i32) {
    // No-op on non-macOS platforms
}

/// Workaround for ONNX Runtime macOS crash on exit.
///
/// This is a workaround for: <https://github.com/microsoft/onnxruntime/issues/24579>
///
/// On macOS with ONNX Runtime 1.21+, the environment cleanup crashes because
/// a static mutex is destroyed before the environment tries to lock it.
///
/// The workaround uses `_exit()` to terminate the process immediately without
/// running C++ destructors or atexit handlers. This prevents the crash but
/// means cleanup code won't run. Since we're at process exit anyway, this is fine.
#[cfg(feature = "ner")]
fn setup_macos_crash_workaround() {
    use std::sync::Once;
    static SETUP_ONCE: Once = Once::new();

    SETUP_ONCE.call_once(|| {
        // On macOS, register an atexit handler that calls _exit() before
        // ONNX Runtime's destructors can run and crash.
        #[cfg(target_os = "macos")]
        {
            unsafe extern "C" {
                fn atexit(func: extern "C" fn()) -> std::ffi::c_int;
            }

            extern "C" fn fast_exit() {
                // Use _exit to skip C++ destructors and atexit handlers.
                // This prevents the ONNX Runtime mutex crash.
                let code = EXIT_CODE.load(std::sync::atomic::Ordering::SeqCst);
                unsafe {
                    libc::_exit(code);
                }
            }

            unsafe {
                atexit(fast_exit);
            }
            log::debug!("Registered macOS ONNX Runtime crash workaround");
        }
    });
}

/// A cached entity for fast repeated lookups
#[cfg(feature = "ner")]
#[derive(Debug, Clone)]
struct CachedEntity {
    text: String,
    label: String,
    confidence: f32,
}

/// NER-based PII detector using GLiNER with smart chunking.
///
/// For long texts, the detector:
/// 1. Splits text into overlapping chunks (default 250 words with 50 word overlap)
/// 2. Runs NER on each chunk
/// 3. Caches discovered entities for fast repeated lookup
/// 4. Uses regex matching to find cached entities in subsequent chunks (faster than NER)
/// 5. Deduplicates and merges results with correct offsets
///
/// # Note on Resource Management
///
/// Due to a known bug in ONNX Runtime 1.21+ on macOS, the NER model is intentionally
/// leaked on drop to prevent a mutex crash during process exit. This is a workaround
/// until the fix is released in a future ONNX Runtime version.
/// See: https://github.com/microsoft/onnxruntime/issues/24579
#[cfg(feature = "ner")]
pub struct NerDetector {
    /// The GLiNER model wrapped in ManuallyDrop to prevent cleanup crash on macOS.
    /// ONNX Runtime 1.21+ has a bug where OrtEnv destruction crashes due to
    /// static mutex destruction order issues.
    model: std::mem::ManuallyDrop<GLiNER<SpanMode>>,
    labels: Vec<String>,
    threshold: f32,
    chunk_size: usize,
    chunk_overlap: usize,
}

#[cfg(feature = "ner")]
impl NerDetector {
    /// Create a new NER detector from model files.
    ///
    /// # Arguments
    /// * `tokenizer_path` - Path to tokenizer.json
    /// * `model_path` - Path to model.onnx
    /// * `labels` - Entity labels to detect (None uses defaults)
    /// * `threshold` - Confidence threshold (0.0-1.0, default 0.5)
    pub fn new(
        tokenizer_path: impl AsRef<Path>,
        model_path: impl AsRef<Path>,
        labels: Option<Vec<String>>,
        threshold: Option<f32>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // IMPORTANT: Set up workaround for macOS ONNX Runtime crash on exit.
        // See: https://github.com/microsoft/onnxruntime/issues/24579
        setup_macos_crash_workaround();

        let model = GLiNER::<SpanMode>::new(
            Parameters::default(),
            RuntimeParameters::default(),
            tokenizer_path.as_ref(),
            model_path.as_ref(),
        )?;

        let labels = labels.unwrap_or_else(|| PII_LABELS.iter().map(|s| s.to_string()).collect());

        Ok(Self {
            // Wrap in ManuallyDrop to prevent ONNX Runtime cleanup crash on macOS.
            // This intentionally leaks the model memory, but prevents the mutex crash.
            model: std::mem::ManuallyDrop::new(model),
            labels,
            threshold: threshold.unwrap_or(0.5),
            chunk_size: DEFAULT_CHUNK_SIZE,
            chunk_overlap: DEFAULT_CHUNK_OVERLAP,
        })
    }

    /// Detect PII entities in text using NER with smart chunking.
    ///
    /// For texts longer than the model's token limit (~384 tokens), this method:
    /// 1. Splits the text into overlapping chunks
    /// 2. Runs NER on each chunk
    /// 3. Caches discovered entities and finds them in subsequent chunks via regex
    /// 4. Merges and deduplicates results
    ///
    /// Returns empty results for very short texts (< 3 chars or < 2 words) to avoid
    /// ONNX runtime reshape errors.
    pub fn detect(
        &self,
        text: &str,
    ) -> Result<Vec<PiiMatch>, Box<dyn std::error::Error + Send + Sync>> {
        // Skip very short text to avoid ONNX reshape errors
        let trimmed = text.trim();
        if trimmed.len() < MIN_TEXT_LENGTH {
            return Ok(Vec::new());
        }

        let word_count = trimmed.split_whitespace().count();
        if word_count < MIN_WORD_COUNT {
            return Ok(Vec::new());
        }

        // For short texts, process directly without chunking
        if word_count <= self.chunk_size {
            return self.detect_chunk(text, 0);
        }

        // Process with chunking for long texts
        self.detect_with_chunking(text)
    }

    /// Detect entities in a single chunk (no chunking).
    fn detect_chunk(
        &self,
        text: &str,
        offset: usize,
    ) -> Result<Vec<PiiMatch>, Box<dyn std::error::Error + Send + Sync>> {
        let label_refs: Vec<&str> = self.labels.iter().map(|s| s.as_str()).collect();

        let input = TextInput::from_str(&[text], &label_refs)?;
        let output = self.model.inference(input)?;

        let mut matches = Vec::new();

        if let Some(spans) = output.spans.first() {
            for span in spans {
                if span.probability() < self.threshold {
                    continue;
                }

                let matched_text = span.text();
                let (start, end) = span.offsets();
                let label = span.class();

                // Filter out low-quality detections
                if !Self::is_valid_entity(label, matched_text) {
                    continue;
                }

                matches.push(PiiMatch {
                    pattern_name: label_to_pattern_name(label).to_string(),
                    matched_text: matched_text.to_string(),
                    start: start + offset,
                    end: end + offset,
                    confidence: Self::probability_to_confidence(span.probability()),
                    category: label_to_category(label),
                });
            }
        }

        Ok(matches)
    }

    /// Check if an entity detection is valid based on type-specific rules.
    ///
    /// This filters out common false positives like:
    /// - Very short "phone numbers" (e.g., "78")
    /// - Single-word locations that are too generic
    fn is_valid_entity(label: &str, text: &str) -> bool {
        let text = text.trim();

        match label.to_lowercase().as_str() {
            // Phone numbers should have at least 7 digits
            "phone_number" | "phone" => {
                let digit_count = text.chars().filter(|c| c.is_ascii_digit()).count();
                digit_count >= 7
            }
            // Persons should have at least 2 characters and ideally a space (first + last)
            "person" => text.len() >= 2,
            // Countries should be at least 2 characters
            "country" => text.len() >= 2,
            // Cities should be at least 2 characters
            "city" => text.len() >= 2,
            // Street addresses should have multiple words or a number
            "street_address" | "address" => {
                text.split_whitespace().count() >= 2 || text.chars().any(|c| c.is_ascii_digit())
            }
            // Organizations should be at least 2 characters
            "organization" | "company" => text.len() >= 2,
            // Default: accept if at least 2 characters
            _ => text.len() >= 2,
        }
    }

    /// Detect entities with smart chunking for long texts.
    fn detect_with_chunking(
        &self,
        text: &str,
    ) -> Result<Vec<PiiMatch>, Box<dyn std::error::Error + Send + Sync>> {
        let chunks = self.split_into_chunks(text);
        let mut all_matches: Vec<PiiMatch> = Vec::new();
        let mut entity_cache: Vec<CachedEntity> = Vec::new();

        for (chunk_text, chunk_offset) in &chunks {
            // First, find cached entities in this chunk using fast string matching
            let cached_matches =
                self.find_cached_entities(chunk_text, *chunk_offset, &entity_cache);
            all_matches.extend(cached_matches);

            // Then run NER on the chunk to find new entities
            let ner_matches = self.detect_chunk(chunk_text, *chunk_offset)?;

            // Add new entities to cache
            for m in &ner_matches {
                let already_cached = entity_cache.iter().any(|e| {
                    e.text.eq_ignore_ascii_case(&m.matched_text) && e.label == m.pattern_name
                });
                if !already_cached && m.matched_text.len() >= 2 {
                    entity_cache.push(CachedEntity {
                        text: m.matched_text.clone(),
                        label: m.pattern_name.clone(),
                        confidence: match m.confidence {
                            Confidence::High => 0.9,
                            Confidence::Medium => 0.7,
                            Confidence::Low => 0.4,
                        },
                    });
                }
            }

            all_matches.extend(ner_matches);
        }

        // Deduplicate overlapping matches (from chunk overlap regions)
        self.deduplicate_matches(&mut all_matches);

        // Sort by position
        all_matches.sort_by_key(|m| (m.start, std::cmp::Reverse(m.end - m.start)));

        Ok(all_matches)
    }

    /// Split text into overlapping chunks, returning (chunk_text, byte_offset) pairs.
    fn split_into_chunks(&self, text: &str) -> Vec<(String, usize)> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut chunks = Vec::new();

        if words.is_empty() {
            return chunks;
        }

        let step = self.chunk_size.saturating_sub(self.chunk_overlap).max(1);
        let mut word_idx = 0;
        let mut byte_offset = 0;

        while word_idx < words.len() {
            let end_idx = (word_idx + self.chunk_size).min(words.len());
            let chunk_words = &words[word_idx..end_idx];
            let chunk_text = chunk_words.join(" ");

            // Calculate the actual byte offset in the original text
            // Find where this chunk starts by locating the first word
            if word_idx > 0 {
                // Find the byte position of the first word of this chunk
                if let Some(pos) = text[byte_offset..].find(chunk_words[0]) {
                    byte_offset += pos;
                }
            }

            chunks.push((chunk_text, byte_offset));

            if end_idx >= words.len() {
                break;
            }

            word_idx += step;
        }

        chunks
    }

    /// Find cached entities in text using fast string matching.
    fn find_cached_entities(
        &self,
        text: &str,
        offset: usize,
        cache: &[CachedEntity],
    ) -> Vec<PiiMatch> {
        let mut matches = Vec::new();
        let text_lower = text.to_lowercase();

        for entity in cache {
            let entity_lower = entity.text.to_lowercase();

            // Find all occurrences of this entity in the text
            let mut search_start = 0;
            while let Some(pos) = text_lower[search_start..].find(&entity_lower) {
                let abs_pos = search_start + pos;

                // Verify word boundaries to avoid partial matches
                let is_word_start =
                    abs_pos == 0 || !text.as_bytes()[abs_pos - 1].is_ascii_alphanumeric();
                let end_pos = abs_pos + entity.text.len();
                let is_word_end =
                    end_pos >= text.len() || !text.as_bytes()[end_pos].is_ascii_alphanumeric();

                if is_word_start && is_word_end {
                    // Extract the actual text (preserving original case)
                    let matched_text = &text[abs_pos..end_pos];

                    matches.push(PiiMatch {
                        pattern_name: entity.label.clone(),
                        matched_text: matched_text.to_string(),
                        start: offset + abs_pos,
                        end: offset + end_pos,
                        confidence: Self::probability_to_confidence(entity.confidence),
                        category: label_to_category(&entity.label),
                    });
                }

                search_start = abs_pos + 1;
            }
        }

        matches
    }

    /// Remove duplicate matches from overlapping chunk regions.
    fn deduplicate_matches(&self, matches: &mut Vec<PiiMatch>) {
        if matches.len() <= 1 {
            return;
        }

        // Sort by (start, end, pattern_name) for consistent ordering
        matches.sort_by(|a, b| {
            a.start
                .cmp(&b.start)
                .then_with(|| a.end.cmp(&b.end))
                .then_with(|| a.pattern_name.cmp(&b.pattern_name))
        });

        // Remove exact duplicates and overlapping matches for the same text
        let mut i = 0;
        while i < matches.len() {
            let mut j = i + 1;
            while j < matches.len() {
                let a = &matches[i];
                let b = &matches[j];

                // Remove if same position and same or similar text
                let is_duplicate = a.start == b.start
                    && a.end == b.end
                    && a.matched_text.eq_ignore_ascii_case(&b.matched_text);

                // Remove if one completely contains the other with same text
                let is_contained = (a.start <= b.start
                    && a.end >= b.end
                    && a.matched_text.eq_ignore_ascii_case(&b.matched_text))
                    || (b.start <= a.start
                        && b.end >= a.end
                        && a.matched_text.eq_ignore_ascii_case(&b.matched_text));

                if is_duplicate || is_contained {
                    // Keep the one with higher confidence
                    if b.confidence > a.confidence {
                        matches.swap(i, j);
                    }
                    matches.remove(j);
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
    }

    /// Convert probability to confidence level.
    fn probability_to_confidence(prob: f32) -> Confidence {
        if prob > 0.8 {
            Confidence::High
        } else if prob > 0.5 {
            Confidence::Medium
        } else {
            Confidence::Low
        }
    }

    /// Get the configured labels.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Public API - used by consumers")
    )]
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Get the confidence threshold.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Public API - used by consumers")
    )]
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Get the chunk size in words.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Public API - used by consumers")
    )]
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Get the chunk overlap in words.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Public API - used by consumers")
    )]
    pub fn chunk_overlap(&self) -> usize {
        self.chunk_overlap
    }
}

/// Default model repository on HuggingFace.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "Public API - used by consumers")
)]
pub const DEFAULT_NER_MODEL: &str = "onnx-community/gliner_multi-v2.1";

/// Model paths configuration for NER.
#[cfg(feature = "ner")]
#[derive(Debug, Clone)]
pub struct NerModelPaths {
    pub tokenizer: std::path::PathBuf,
    pub model: std::path::PathBuf,
}

#[cfg(feature = "ner")]
impl NerModelPaths {
    /// Create from explicit paths.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Public API - used by consumers")
    )]
    pub fn new(tokenizer: std::path::PathBuf, model: std::path::PathBuf) -> Self {
        Self { tokenizer, model }
    }

    /// Check if model files exist.
    pub fn exists(&self) -> bool {
        self.tokenizer.exists() && self.model.exists()
    }
}

/// NER model configuration with cache and download support.
#[cfg(feature = "ner")]
#[derive(Debug, Clone)]
pub struct NerModelConfig {
    /// HuggingFace model repository (e.g., "onnx-community/gliner_multi-v2.1")
    pub model_repo: String,
    /// Custom cache directory (None = use HF_HOME or default ~/.cache/huggingface/)
    pub cache_dir: Option<std::path::PathBuf>,
}

#[cfg(feature = "ner")]
impl Default for NerModelConfig {
    fn default() -> Self {
        Self {
            model_repo: DEFAULT_NER_MODEL.to_string(),
            cache_dir: None,
        }
    }
}

#[cfg(feature = "ner")]
impl NerModelConfig {
    /// Create with a custom model repository.
    pub fn with_model(model_repo: impl Into<String>) -> Self {
        Self {
            model_repo: model_repo.into(),
            cache_dir: None,
        }
    }

    /// Set a custom cache directory.
    pub fn with_cache_dir(mut self, cache_dir: impl Into<std::path::PathBuf>) -> Self {
        self.cache_dir = Some(cache_dir.into());
        self
    }

    /// Download model files and return paths.
    ///
    /// Uses HuggingFace Hub cache. Respects:
    /// - `HF_HOME` env var for cache location
    /// - `HF_ENDPOINT` env var for custom endpoint
    /// - Custom cache_dir if set in config
    ///
    /// Files are cached and reused on subsequent calls.
    pub fn download(&self) -> Result<NerModelPaths, Box<dyn std::error::Error + Send + Sync>> {
        use hf_hub::api::sync::ApiBuilder;

        // Build API with optional custom cache dir
        let api = if let Some(ref cache_dir) = self.cache_dir {
            ApiBuilder::new()
                .with_cache_dir(cache_dir.clone())
                .build()?
        } else {
            // Use HF_HOME env var or default
            ApiBuilder::from_env().build()?
        };

        let repo = api.model(self.model_repo.clone());

        // Download tokenizer.json
        let tokenizer_path = repo.get("tokenizer.json")?;

        // Download ONNX model
        let model_path = repo.get("onnx/model.onnx")?;

        Ok(NerModelPaths {
            tokenizer: tokenizer_path,
            model: model_path,
        })
    }

    /// Check if model is already cached.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Public API - used by consumers")
    )]
    pub fn is_cached(&self) -> bool {
        // Try to get paths without downloading
        if let Ok(paths) = self.get_cached_paths() {
            paths.exists()
        } else {
            false
        }
    }

    /// Get cached paths without downloading.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Public API - used by consumers")
    )]
    fn get_cached_paths(&self) -> Result<NerModelPaths, Box<dyn std::error::Error + Send + Sync>> {
        use hf_hub::Cache;

        let cache = if let Some(ref cache_dir) = self.cache_dir {
            Cache::new(cache_dir.clone())
        } else {
            Cache::default()
        };

        let repo = cache.model(self.model_repo.clone());

        // Check if files exist in cache
        let tokenizer_path = repo
            .get("tokenizer.json")
            .ok_or("tokenizer.json not in cache")?;
        let model_path = repo
            .get("onnx/model.onnx")
            .ok_or("onnx/model.onnx not in cache")?;

        Ok(NerModelPaths {
            tokenizer: tokenizer_path,
            model: model_path,
        })
    }
}

#[cfg(all(test, feature = "ner"))]
mod tests {
    use super::*;
    use crate::engine::patterns::PiiCategory;

    #[test]
    fn test_label_mapping() {
        assert_eq!(label_to_pattern_name("person"), "person");
        assert_eq!(label_to_pattern_name("Person"), "person");
        assert_eq!(label_to_pattern_name("organization"), "organization");
        assert_eq!(label_to_pattern_name("street_address"), "street_address");
    }

    #[test]
    fn test_category_mapping() {
        assert_eq!(label_to_category("person"), PiiCategory::Identity);
        assert_eq!(label_to_category("city"), PiiCategory::Contact);
        assert_eq!(label_to_category("organization"), PiiCategory::Other);
    }

    #[test]
    fn test_ner_model_paths() {
        let paths = NerModelPaths::new(
            std::path::PathBuf::from("/tmp/tokenizer.json"),
            std::path::PathBuf::from("/tmp/model.onnx"),
        );
        assert!(paths.tokenizer.to_string_lossy().contains("tokenizer.json"));
        assert!(paths.model.to_string_lossy().contains("model.onnx"));
    }

    #[test]
    fn test_default_ner_model() {
        assert_eq!(DEFAULT_NER_MODEL, "onnx-community/gliner_multi-v2.1");
    }
}
