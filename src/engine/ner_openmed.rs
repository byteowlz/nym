//! OpenMed token-classification NER backend.
//!
//! This module provides a second, parallel NER backend alongside the GLiNER
//! span model in [`super::ner`]. Where GLiNER is a zero-shot *span* model loaded
//! through the `gline-rs` crate, OpenMed ships fine-tuned **token classification**
//! models (DeBERTa-v2) with a fixed, rich PII taxonomy (106 BIO labels covering
//! names, emails, SSNs, credit cards, medical record numbers, API keys, and more).
//!
//! Because `gline-rs` only understands GLiNER span models, this backend talks to
//! ONNX Runtime (`ort`) directly:
//!
//! 1. Tokenize with the HF `tokenizers` crate (offsets enabled).
//! 2. Run `input_ids` + `attention_mask` through the ONNX session.
//! 3. Argmax + softmax over the per-token label logits.
//! 4. BIO-decode contiguous tokens into entity spans.
//! 5. Map sub-word offsets back to byte offsets in the original text.
//!
//! A converted model directory is produced by `scripts/convert_openmed_onnx.sh`
//! and must contain `model.onnx`, `tokenizer.json`, and `config.json`.

#![cfg(feature = "ner")]

use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::engine::detector::PiiMatch;
use crate::engine::patterns::{Confidence, PiiCategory};

/// Default OpenMed model when the backend is selected but none is configured.
/// A HuggingFace repo id (with subfolder) that nym downloads + caches on first
/// use; the small int8 model is the best speed/accuracy/memory trade-off.
pub const DEFAULT_OPENMED_MODEL: &str = "Wismut/openmed-onnx/small";

/// Default chunk size in words. The model has a 512-token limit; this keeps a
/// conservative margin for sub-word expansion plus special tokens.
const DEFAULT_CHUNK_SIZE: usize = 200;

/// Default overlap in words to avoid splitting entities across chunk boundaries.
const DEFAULT_CHUNK_OVERLAP: usize = 40;

/// Minimum text length in characters before attempting inference.
const MIN_TEXT_LENGTH: usize = 3;

/// A decoded entity span within a single chunk, in chunk-local byte offsets.
struct DecodedSpan {
    base_label: String,
    start: usize,
    end: usize,
    prob_sum: f32,
    token_count: usize,
}

/// OpenMed DeBERTa-v2 token-classification NER backend.
pub struct OpenMedDetector {
    /// ONNX Runtime session. Wrapped in `ManuallyDrop` to match the GLiNER
    /// backend's workaround for the ONNX Runtime macOS exit crash.
    /// See [`super::ner`] for details.
    session: std::mem::ManuallyDrop<Session>,
    tokenizer: Tokenizer,
    /// Maps a label index to its BIO label string (e.g. `"B-email"`).
    id2label: Vec<String>,
    threshold: f32,
    chunk_size: usize,
    chunk_overlap: usize,
}

impl OpenMedDetector {
    /// Create a detector from a converted model directory.
    ///
    /// The directory must contain `tokenizer.json`, `config.json`, and an ONNX
    /// model (as produced by `scripts/convert_openmed_onnx.sh`). If a quantized
    /// `model_int8.onnx` is present it is preferred over the fp32 `model.onnx`.
    pub fn from_dir(
        model_dir: impl AsRef<Path>,
        threshold: Option<f32>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let dir = model_dir.as_ref();
        let quantized = dir.join("model_int8.onnx");
        let model_path = if quantized.exists() {
            quantized
        } else {
            dir.join("model.onnx")
        };
        Self::build(&model_path, &dir.join("tokenizer.json"), &dir.join("config.json"), threshold)
    }

    /// Create a detector by downloading a converted model from a HuggingFace
    /// repository (cached via `hf-hub`, like the GLiNER backend).
    ///
    /// `model_ref` is either a repo id (`org/name`) or a repo id with a
    /// subfolder (`org/name/subdir`), so several models can live in one repo.
    /// The (sub)folder must contain `tokenizer.json`, `config.json`, and
    /// `model.onnx` (and optionally `model_int8.onnx`, preferred when present).
    /// Respects `HF_HOME`/`HF_ENDPOINT`; `cache_dir` overrides the cache location.
    pub fn from_repo(
        model_ref: &str,
        cache_dir: Option<&Path>,
        threshold: Option<f32>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        use hf_hub::api::sync::ApiBuilder;

        // Split "org/name[/sub/dir]" into the repo id and an optional subfolder.
        let parts: Vec<&str> = model_ref.split('/').collect();
        let (repo, prefix) = if parts.len() > 2 {
            (parts[..2].join("/"), format!("{}/", parts[2..].join("/")))
        } else {
            (model_ref.to_string(), String::new())
        };

        let api = match cache_dir {
            Some(dir) => ApiBuilder::new().with_cache_dir(dir.to_path_buf()).build()?,
            None => ApiBuilder::from_env().build()?,
        };
        let model = api.model(repo);

        let config_path = model.get(&format!("{prefix}config.json"))?;
        let tokenizer_path = model.get(&format!("{prefix}tokenizer.json"))?;
        // Prefer the quantized model if the (sub)folder publishes one.
        let model_path = match model.get(&format!("{prefix}model_int8.onnx")) {
            Ok(p) => p,
            Err(_) => model.get(&format!("{prefix}model.onnx"))?,
        };

        Self::build(&model_path, &tokenizer_path, &config_path, threshold)
    }

    /// Build a detector from explicit file paths.
    fn build(
        model_path: &Path,
        tokenizer_path: &Path,
        config_path: &Path,
        threshold: Option<f32>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let id2label = load_id2label(config_path)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;
        let session = Session::builder()?.commit_from_file(model_path)?;

        Ok(Self {
            session: std::mem::ManuallyDrop::new(session),
            tokenizer,
            id2label,
            threshold: threshold.unwrap_or(0.5),
            chunk_size: DEFAULT_CHUNK_SIZE,
            chunk_overlap: DEFAULT_CHUNK_OVERLAP,
        })
    }

    /// Detect PII entities in `text`, chunking long inputs.
    pub fn detect(
        &self,
        text: &str,
    ) -> Result<Vec<PiiMatch>, Box<dyn std::error::Error + Send + Sync>> {
        let trimmed = text.trim();
        if trimmed.len() < MIN_TEXT_LENGTH {
            return Ok(Vec::new());
        }

        let word_count = trimmed.split_whitespace().count();
        if word_count <= self.chunk_size {
            return self.detect_chunk(text, 0);
        }

        let mut all_matches = Vec::new();
        for (chunk_text, chunk_offset) in self.split_into_chunks(text) {
            let matches = self.detect_chunk(&chunk_text, chunk_offset)?;
            all_matches.extend(matches);
        }
        self.deduplicate_matches(&mut all_matches);
        all_matches.sort_by_key(|m| (m.start, std::cmp::Reverse(m.end - m.start)));
        Ok(all_matches)
    }

    /// Run inference on a single chunk and decode entities at `offset`.
    fn detect_chunk(
        &self,
        text: &str,
        offset: usize,
    ) -> Result<Vec<PiiMatch>, Box<dyn std::error::Error + Send + Sync>> {
        let encoding = self.tokenizer.encode(text, true)?;
        let ids = encoding.get_ids();
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let seq_len = ids.len();
        let num_labels = self.id2label.len();

        let input_ids: Vec<i64> = ids.iter().map(|&id| i64::from(id)).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| i64::from(m))
            .collect();

        let shape = vec![1_i64, seq_len as i64];
        let ids_tensor = Tensor::from_array((shape.clone(), input_ids))?;
        let mask_tensor = Tensor::from_array((shape, attention_mask))?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor,
        ]?)?;

        let (_, logits) = outputs["logits"].try_extract_raw_tensor::<f32>()?;
        if logits.len() != seq_len * num_labels {
            return Err(format!(
                "unexpected logits length: got {}, expected {}",
                logits.len(),
                seq_len * num_labels
            )
            .into());
        }

        let offsets = encoding.get_offsets();
        let special_mask = encoding.get_special_tokens_mask();
        let spans = self.decode_bio(logits, num_labels, offsets, special_mask);

        let mut matches = Vec::new();
        for span in spans {
            // Trim leading/trailing whitespace that sub-word offsets often include.
            let (start, end) = trim_span(text, span.start, span.end);
            if start >= end {
                continue;
            }
            let matched_text = &text[start..end];
            let avg_prob = if span.token_count > 0 {
                span.prob_sum / span.token_count as f32
            } else {
                0.0
            };
            let (pattern_name, category) = label_to_pattern(&span.base_label);
            if !is_valid_entity(&span.base_label, matched_text) {
                continue;
            }
            matches.push(PiiMatch {
                pattern_name: pattern_name.to_string(),
                matched_text: matched_text.to_string(),
                start: start + offset,
                end: end + offset,
                confidence: probability_to_confidence(avg_prob),
                category,
            });
        }
        Ok(matches)
    }

    /// BIO-decode per-token predictions into contiguous entity spans.
    fn decode_bio(
        &self,
        logits: &[f32],
        num_labels: usize,
        offsets: &[(usize, usize)],
        special_mask: &[u32],
    ) -> Vec<DecodedSpan> {
        let mut spans: Vec<DecodedSpan> = Vec::new();
        let mut current: Option<DecodedSpan> = None;

        for (token_idx, &(tok_start, tok_end)) in offsets.iter().enumerate() {
            // Skip special tokens ([CLS], [SEP], padding) and empty offsets.
            if special_mask.get(token_idx).copied().unwrap_or(0) == 1 {
                continue;
            }
            if tok_start == tok_end {
                continue;
            }

            let row = &logits[token_idx * num_labels..(token_idx + 1) * num_labels];
            let (best_idx, prob) = argmax_softmax(row);
            let label = self.id2label.get(best_idx).map_or("O", String::as_str);

            if label == "O" || prob < self.threshold {
                if let Some(span) = current.take() {
                    spans.push(span);
                }
                continue;
            }

            let (prefix, base) = split_bio(label);

            match current.as_mut() {
                // Continue the current entity only for an `I-` of the same base type.
                Some(span) if prefix == BioPrefix::Inside && span.base_label == base => {
                    span.end = tok_end;
                    span.prob_sum += prob;
                    span.token_count += 1;
                }
                _ => {
                    if let Some(span) = current.take() {
                        spans.push(span);
                    }
                    current = Some(DecodedSpan {
                        base_label: base.to_string(),
                        start: tok_start,
                        end: tok_end,
                        prob_sum: prob,
                        token_count: 1,
                    });
                }
            }
        }

        if let Some(span) = current.take() {
            spans.push(span);
        }
        spans
    }

    /// Split text into overlapping word chunks, returning `(chunk, byte_offset)`.
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

            if word_idx > 0 {
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

    /// Remove duplicate/contained matches arising from chunk overlap.
    fn deduplicate_matches(&self, matches: &mut Vec<PiiMatch>) {
        if matches.len() <= 1 {
            return;
        }
        matches.sort_by(|a, b| {
            a.start
                .cmp(&b.start)
                .then_with(|| a.end.cmp(&b.end))
                .then_with(|| a.pattern_name.cmp(&b.pattern_name))
        });

        let mut i = 0;
        while i < matches.len() {
            let mut j = i + 1;
            while j < matches.len() {
                let same_span = matches[i].start == matches[j].start
                    && matches[i].end == matches[j].end
                    && matches[i]
                        .matched_text
                        .eq_ignore_ascii_case(&matches[j].matched_text);
                if same_span {
                    if matches[j].confidence > matches[i].confidence {
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

    /// Get the confidence threshold.
    #[cfg_attr(not(test), expect(dead_code, reason = "Public API - used by consumers"))]
    pub fn threshold(&self) -> f32 {
        self.threshold
    }
}

/// BIO prefix of a label.
#[derive(PartialEq, Eq)]
enum BioPrefix {
    Begin,
    Inside,
    Other,
}

/// Split a BIO label such as `"B-email"` into its prefix and base type.
fn split_bio(label: &str) -> (BioPrefix, &str) {
    if let Some(rest) = label.strip_prefix("B-") {
        (BioPrefix::Begin, rest)
    } else if let Some(rest) = label.strip_prefix("I-") {
        (BioPrefix::Inside, rest)
    } else {
        (BioPrefix::Other, label)
    }
}

/// Numerically stable softmax over `row`, returning `(argmax_index, max_prob)`.
fn argmax_softmax(row: &[f32]) -> (usize, f32) {
    let mut best_idx = 0;
    let mut max = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > max {
            max = v;
            best_idx = i;
        }
    }
    let sum: f32 = row.iter().map(|&v| (v - max).exp()).sum();
    let prob = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    (best_idx, prob)
}

/// Trim leading/trailing ASCII whitespace from a byte span of `text`.
fn trim_span(text: &str, mut start: usize, mut end: usize) -> (usize, usize) {
    let bytes = text.as_bytes();
    while start < end && bytes[start].is_ascii_whitespace() {
        start += 1;
    }
    while end > start && bytes[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    (start, end)
}

/// Convert a probability to a [`Confidence`] level (matches the GLiNER backend).
fn probability_to_confidence(prob: f32) -> Confidence {
    if prob > 0.8 {
        Confidence::High
    } else if prob > 0.5 {
        Confidence::Medium
    } else {
        Confidence::Low
    }
}

/// Type-specific validity filtering to suppress obvious false positives.
fn is_valid_entity(base_label: &str, text: &str) -> bool {
    let text = text.trim();
    match base_label {
        "phone_number" | "fax_number" => {
            text.chars().filter(char::is_ascii_digit).count() >= 7
        }
        "ssn" | "credit_debit_card" => {
            text.chars().filter(char::is_ascii_digit).count() >= 4
        }
        _ => text.chars().count() >= 2,
    }
}

/// Map an OpenMed entity base label to a nym pattern name and category.
///
/// Where an OpenMed label overlaps a built-in nym pattern, the canonical nym
/// name is used so existing placeholder/fake replacement logic applies.
fn label_to_pattern(base: &str) -> (&'static str, PiiCategory) {
    match base {
        // Names / identity
        "first_name" => ("first_name", PiiCategory::Identity),
        "last_name" => ("last_name", PiiCategory::Identity),
        "user_name" => ("username", PiiCategory::Social),
        "age" => ("age", PiiCategory::Identity),
        "gender" => ("gender", PiiCategory::Identity),
        "date_of_birth" => ("date_of_birth", PiiCategory::Identity),
        // Government / record identifiers
        "ssn" => ("ssn", PiiCategory::Identity),
        "tax_id" => ("tax_id", PiiCategory::Identity),
        "medical_record_number" => ("medical_record_number", PiiCategory::Identity),
        "health_plan_beneficiary_number" => {
            ("health_plan_beneficiary_number", PiiCategory::Identity)
        }
        "certificate_license_number" => ("certificate_license_number", PiiCategory::Identity),
        "account_number" => ("account_number", PiiCategory::Identity),
        "customer_id" | "employee_id" | "unique_id" | "device_identifier" => {
            ("unique_id", PiiCategory::Identity)
        }
        "biometric_identifier" => ("biometric_identifier", PiiCategory::Identity),
        // Contact
        "email" => ("email", PiiCategory::Contact),
        "phone_number" => ("phone_ner", PiiCategory::Contact),
        "fax_number" => ("fax_number", PiiCategory::Contact),
        // Location
        "street_address" => ("street_address", PiiCategory::Contact),
        "city" => ("city", PiiCategory::Contact),
        "county" => ("county", PiiCategory::Contact),
        "state" => ("state", PiiCategory::Contact),
        "country" => ("country", PiiCategory::Contact),
        "postcode" => ("postcode", PiiCategory::Contact),
        "coordinate" => ("coordinate", PiiCategory::Network),
        // Financial
        "credit_debit_card" => ("credit_card", PiiCategory::Financial),
        "cvv" => ("cvv", PiiCategory::Financial),
        "pin" => ("pin", PiiCategory::Financial),
        "bank_routing_number" => ("bank_routing_number", PiiCategory::Financial),
        "swift_bic" => ("swift_bic", PiiCategory::Financial),
        // Network / technical
        "ipv4" => ("ipv4", PiiCategory::Network),
        "ipv6" => ("ipv6", PiiCategory::Network),
        "mac_address" => ("mac", PiiCategory::Network),
        "url" => ("url", PiiCategory::Network),
        "http_cookie" => ("http_cookie", PiiCategory::Network),
        // Credentials
        "api_key" => ("api_key", PiiCategory::Authentication),
        "password" => ("password", PiiCategory::Authentication),
        // Organization
        "company_name" => ("organization", PiiCategory::Other),
        // Temporal
        "date" | "date_time" => ("date", PiiCategory::Other),
        "time" => ("time", PiiCategory::Other),
        // Vehicle
        "license_plate" => ("license_plate", PiiCategory::Other),
        "vehicle_identifier" => ("vehicle_identifier", PiiCategory::Other),
        // Sensitive demographics and free-form fields
        _ => ("openmed_entity", PiiCategory::Other),
    }
}

/// Load the `id2label` map from a model `config.json` into an index-ordered Vec.
fn load_id2label(
    config_path: &Path,
) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
    let raw = std::fs::read_to_string(config_path)?;
    let json: serde_json::Value = serde_json::from_str(&raw)?;
    let map = json
        .get("id2label")
        .and_then(|v| v.as_object())
        .ok_or("config.json missing id2label object")?;

    let mut pairs: Vec<(usize, String)> = Vec::with_capacity(map.len());
    for (k, v) in map {
        let idx: usize = k.parse()?;
        let label = v
            .as_str()
            .ok_or("id2label value is not a string")?
            .to_string();
        pairs.push((idx, label));
    }
    pairs.sort_by_key(|(idx, _)| *idx);

    // Ensure indices are contiguous from 0.
    let mut labels = Vec::with_capacity(pairs.len());
    for (expected, (idx, label)) in pairs.into_iter().enumerate() {
        if idx != expected {
            return Err(format!("id2label is not contiguous at index {expected}").into());
        }
        labels.push(label);
    }
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_bio() {
        assert!(matches!(split_bio("B-email"), (BioPrefix::Begin, "email")));
        assert!(matches!(split_bio("I-ssn"), (BioPrefix::Inside, "ssn")));
        assert!(matches!(split_bio("O"), (BioPrefix::Other, "O")));
    }

    #[test]
    fn test_argmax_softmax() {
        let (idx, prob) = argmax_softmax(&[0.0, 5.0, 1.0]);
        assert_eq!(idx, 1);
        assert!(prob > 0.9);
    }

    #[test]
    fn test_trim_span() {
        let text = "  John  ";
        let (s, e) = trim_span(text, 0, text.len());
        assert_eq!(&text[s..e], "John");
    }

    #[test]
    fn test_label_to_pattern() {
        assert_eq!(label_to_pattern("email").0, "email");
        assert_eq!(label_to_pattern("credit_debit_card").0, "credit_card");
        assert_eq!(label_to_pattern("phone_number").0, "phone_ner");
        assert_eq!(label_to_pattern("unknown_thing").0, "openmed_entity");
    }

    #[test]
    fn test_is_valid_entity() {
        assert!(!is_valid_entity("phone_number", "12"));
        assert!(is_valid_entity("phone_number", "415-555-7012"));
        assert!(is_valid_entity("first_name", "Jo"));
        assert!(!is_valid_entity("first_name", "X"));
    }
}
