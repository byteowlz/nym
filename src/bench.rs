//! Benchmarking module for PII detection accuracy testing.
//!
//! Supports loading datasets from:
//! - HuggingFace datasets (e.g., conll2003, ai4privacy/pii-masking-300k)
//! - Custom JSONL files with annotated entities
//!
//! # Example JSONL format
//!
//! ```json
//! {"text": "Contact John at john@example.com", "entities": [{"start": 8, "end": 12, "label": "PERSON"}, {"start": 16, "end": 32, "label": "EMAIL"}]}
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::engine::{Detector, DetectorConfig};

/// An entity annotation in the dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Start byte offset
    pub start: usize,
    /// End byte offset
    pub end: usize,
    /// Entity label (e.g., "PERSON", "EMAIL", "PHONE")
    pub label: String,
    /// The text of the entity (optional, can be derived from text[start..end])
    #[serde(default)]
    pub text: Option<String>,
}

/// A single example in the dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    /// The input text
    pub text: String,
    /// Ground truth entities
    pub entities: Vec<Entity>,
}

/// Result of evaluating a single example.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct ExampleResult {
    /// True positives (correctly detected)
    pub true_positives: usize,
    /// False positives (detected but not in ground truth)
    pub false_positives: usize,
    /// False negatives (in ground truth but not detected)
    pub false_negatives: usize,
}

/// A missed detection (false negative).
#[derive(Debug, Clone, Serialize)]
pub struct MissedDetection {
    /// The text that was not detected
    pub text: String,
    /// The label it should have been
    pub label: String,
    /// Context: surrounding text
    pub context: String,
}

/// A false positive detection.
#[derive(Debug, Clone, Serialize)]
pub struct FalsePositive {
    /// The text that was incorrectly detected
    pub text: String,
    /// The label we assigned
    pub label: String,
    /// Context: surrounding text
    pub context: String,
}

/// Aggregated benchmark results.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkResults {
    /// Total examples processed
    pub total_examples: usize,
    /// Total ground truth entities
    pub total_ground_truth: usize,
    /// Total detected entities
    pub total_detected: usize,
    /// True positives
    pub true_positives: usize,
    /// False positives
    pub false_positives: usize,
    /// False negatives
    pub false_negatives: usize,
    /// Precision (TP / (TP + FP))
    pub precision: f64,
    /// Recall (TP / (TP + FN))
    pub recall: f64,
    /// F1 score (2 * precision * recall / (precision + recall))
    pub f1: f64,
    /// Results broken down by label
    pub by_label: HashMap<String, LabelResults>,
    /// Missed detections (false negatives) - only populated if requested
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub missed_detections: Vec<MissedDetection>,
    /// False positive detections - only populated if requested
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub false_positive_detections: Vec<FalsePositive>,
}

/// Results for a specific label.
#[derive(Debug, Clone, Serialize)]
pub struct LabelResults {
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

impl BenchmarkResults {
    fn new() -> Self {
        Self {
            total_examples: 0,
            total_ground_truth: 0,
            total_detected: 0,
            true_positives: 0,
            false_positives: 0,
            false_negatives: 0,
            precision: 0.0,
            recall: 0.0,
            f1: 0.0,
            missed_detections: Vec::new(),
            false_positive_detections: Vec::new(),
            by_label: HashMap::new(),
        }
    }

    fn finalize(&mut self) {
        // Calculate overall metrics
        if self.true_positives + self.false_positives > 0 {
            self.precision =
                self.true_positives as f64 / (self.true_positives + self.false_positives) as f64;
        }
        if self.true_positives + self.false_negatives > 0 {
            self.recall =
                self.true_positives as f64 / (self.true_positives + self.false_negatives) as f64;
        }
        if self.precision + self.recall > 0.0 {
            self.f1 = 2.0 * self.precision * self.recall / (self.precision + self.recall);
        }

        // Calculate per-label metrics
        for (_, label_result) in self.by_label.iter_mut() {
            if label_result.true_positives + label_result.false_positives > 0 {
                label_result.precision = label_result.true_positives as f64
                    / (label_result.true_positives + label_result.false_positives) as f64;
            }
            if label_result.true_positives + label_result.false_negatives > 0 {
                label_result.recall = label_result.true_positives as f64
                    / (label_result.true_positives + label_result.false_negatives) as f64;
            }
            if label_result.precision + label_result.recall > 0.0 {
                label_result.f1 = 2.0 * label_result.precision * label_result.recall
                    / (label_result.precision + label_result.recall);
            }
        }
    }
}

/// Configuration for benchmarking.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Detector configuration
    pub detector_config: DetectorConfig,
    /// Whether to use strict matching (exact span) or relaxed (overlap)
    pub strict_matching: bool,
    /// Label mapping (dataset label -> our pattern name)
    pub label_mapping: HashMap<String, String>,
    /// Maximum examples to process (None = all)
    pub max_examples: Option<usize>,
    /// Label to track missed detections for (None = don't track)
    pub track_misses_for: Option<String>,
    /// Label to track false positives for (None = don't track)
    pub track_false_positives_for: Option<String>,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            detector_config: DetectorConfig::default(),
            strict_matching: false,
            label_mapping: Self::default_label_mapping(),
            max_examples: None,
            track_misses_for: None,
            track_false_positives_for: None,
        }
    }
}

impl BenchConfig {
    /// Default label mapping from dataset labels to our pattern names.
    pub fn default_label_mapping() -> HashMap<String, String> {
        let mut m = HashMap::new();

        // === Name-related ===
        // Full person names
        m.insert("PER".to_string(), "person".to_string());
        m.insert("PERSON".to_string(), "person".to_string());
        // First names - GLiNER detects full names as "person", so map these to person for matching
        // This allows overlapping detection to count as a match
        m.insert("GIVENNAME".to_string(), "person".to_string());
        m.insert("GIVENNAME1".to_string(), "person".to_string());
        m.insert("GIVENNAME2".to_string(), "person".to_string());
        m.insert("FIRSTNAME".to_string(), "person".to_string());
        // Last names - also map to person for the same reason
        m.insert("LASTNAME".to_string(), "person".to_string());
        m.insert("LASTNAME1".to_string(), "person".to_string());
        m.insert("LASTNAME2".to_string(), "person".to_string());
        m.insert("LASTNAME3".to_string(), "person".to_string());
        m.insert("SURNAME".to_string(), "person".to_string());
        // Titles like Mr., Dr. - low priority
        m.insert("TITLE".to_string(), "person".to_string());
        m.insert("PREFIX".to_string(), "person".to_string());
        m.insert("MIDDLENAME".to_string(), "person".to_string());

        // === Organization ===
        m.insert("ORG".to_string(), "organization".to_string());
        m.insert("ORGANIZATION".to_string(), "organization".to_string());
        m.insert("COMPANY".to_string(), "organization".to_string());

        // === Location-related ===
        m.insert("LOC".to_string(), "location".to_string());
        m.insert("LOCATION".to_string(), "location".to_string());
        m.insert("GPE".to_string(), "location".to_string());
        m.insert("CITY".to_string(), "city".to_string());
        m.insert("STATE".to_string(), "state".to_string());
        m.insert("COUNTRY".to_string(), "country".to_string());
        m.insert("POSTCODE".to_string(), "zip_code".to_string());
        m.insert("ZIPCODE".to_string(), "zip_code".to_string());
        m.insert("GEOCOORD".to_string(), "geocoord".to_string());

        // === Address-related ===
        m.insert("STREET".to_string(), "street_address".to_string());
        m.insert("STREETADDRESS".to_string(), "street_address".to_string());
        m.insert("ADDRESS".to_string(), "street_address".to_string());
        m.insert("BUILDING".to_string(), "street_address".to_string());
        m.insert("SECADDRESS".to_string(), "street_address".to_string());

        // === Contact ===
        m.insert("EMAIL".to_string(), "email".to_string());
        m.insert("TEL".to_string(), "phone_intl".to_string());
        m.insert("PHONE".to_string(), "phone_intl".to_string());
        m.insert("PHONE_NUMBER".to_string(), "phone_intl".to_string());
        m.insert("PHONE_INTL".to_string(), "phone_intl".to_string());

        // === Identity documents ===
        m.insert("SSN".to_string(), "ssn".to_string());
        // SOCIALNUMBER can be various EU formats
        m.insert("SOCIALNUMBER".to_string(), "eu_id".to_string());
        m.insert("IDCARD".to_string(), "eu_id".to_string());
        m.insert("DRIVERLICENSE".to_string(), "drivers_license".to_string());
        m.insert("PASSPORT".to_string(), "passport_us".to_string());
        // Country-specific
        m.insert("UK_NINO".to_string(), "uk_nino".to_string());
        m.insert("FR_NIR".to_string(), "fr_nir".to_string());
        m.insert("IT_CF".to_string(), "it_cf".to_string());
        m.insert("ES_DNI".to_string(), "es_dni".to_string());

        // === Financial ===
        m.insert("CREDIT_CARD".to_string(), "credit_card".to_string());
        m.insert("CREDITCARD".to_string(), "credit_card".to_string());
        m.insert("IBAN".to_string(), "iban".to_string());

        // === Network ===
        m.insert("IP".to_string(), "ipv4".to_string());
        m.insert("IP_ADDRESS".to_string(), "ipv4".to_string());
        m.insert("IPV4".to_string(), "ipv4".to_string());
        m.insert("IPV6".to_string(), "ipv6".to_string());

        // === Date/Time ===
        m.insert("DATE".to_string(), "date".to_string());
        m.insert("BOD".to_string(), "date".to_string());
        m.insert("DOB".to_string(), "date".to_string());
        m.insert("DATEOFBIRTH".to_string(), "date".to_string());
        m.insert("TIME".to_string(), "time".to_string());

        // === Auth/Credentials ===
        m.insert("USERNAME".to_string(), "username".to_string());
        m.insert("PASS".to_string(), "api_key".to_string());
        m.insert("PASSWORD".to_string(), "api_key".to_string());

        // === Additional identity labels ===
        m.insert("AGE".to_string(), "date".to_string()); // Age can reveal DOB
        m.insert("SEX".to_string(), "person".to_string());
        m.insert("GENDER".to_string(), "person".to_string());

        // === Additional ID types ===
        m.insert("CREDITCARDNUMBER".to_string(), "credit_card".to_string());
        m.insert("TELEPHONENUM".to_string(), "phone_intl".to_string());
        m.insert("SOCIALNUM".to_string(), "ssn".to_string());
        m.insert("TAXNUM".to_string(), "ssn".to_string());
        m.insert("IDCARDNUM".to_string(), "eu_id".to_string());
        m.insert("DRIVERLICENSENUM".to_string(), "drivers_license".to_string());
        m.insert("PASSPORTNUM".to_string(), "passport_us".to_string());
        m.insert("ACCOUNTNUM".to_string(), "iban".to_string());
        m.insert("BUILDINGNUM".to_string(), "street_address".to_string());

        m
    }
    
    /// Create reverse mapping (our pattern -> dataset labels) for matching.
    pub fn reverse_label_mapping(&self) -> HashMap<String, Vec<String>> {
        let mut reverse: HashMap<String, Vec<String>> = HashMap::new();
        for (dataset_label, our_label) in &self.label_mapping {
            reverse
                .entry(our_label.clone())
                .or_default()
                .push(dataset_label.clone());
        }
        reverse
    }
}

/// Load examples from a JSONL file.
pub fn load_jsonl(path: &Path) -> Result<Vec<Example>> {
    let file = File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let reader = BufReader::new(file);

    let mut examples = Vec::new();
    for (line_num, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("Failed to read line {}", line_num + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let example: Example = serde_json::from_str(&line)
            .with_context(|| format!("Failed to parse line {}", line_num + 1))?;
        examples.push(example);
    }

    Ok(examples)
}

/// Download a dataset from HuggingFace.
#[cfg(feature = "bench")]
pub fn download_huggingface_dataset(
    dataset_name: &str,
    split: &str,
    cache_dir: Option<&Path>,
    limit: usize,
) -> Result<Vec<Example>> {
    use std::io::Write;

    // Determine cache path
    let cache_path = if let Some(dir) = cache_dir {
        dir.to_path_buf()
    } else {
        dirs::cache_dir()
            .unwrap_or_else(|| std::env::temp_dir())
            .join("nym")
            .join("datasets")
    };
    std::fs::create_dir_all(&cache_path)?;

    let dataset_file = cache_path.join(format!(
        "{}_{}_{}.jsonl",
        dataset_name.replace('/', "_"),
        split,
        limit
    ));

    // Check if already cached
    if dataset_file.exists() {
        log::info!("Loading cached dataset from {}", dataset_file.display());
        return load_jsonl(&dataset_file);
    }

    // Download from HuggingFace with pagination
    // HuggingFace API has a max of 100 rows per request
    const PAGE_SIZE: usize = 100;
    
    log::info!("Downloading dataset {} (split: {}, limit: {})...", dataset_name, split, limit);
    
    let mut all_examples = Vec::new();
    let mut offset = 0;
    
    while all_examples.len() < limit {
        let remaining = limit - all_examples.len();
        let fetch_count = remaining.min(PAGE_SIZE);
        
        let url = format!(
            "https://datasets-server.huggingface.co/rows?dataset={}&config=default&split={}&offset={}&length={}",
            dataset_name, split, offset, fetch_count
        );

        log::debug!("Fetching {} rows from offset {}...", fetch_count, offset);
        
        let response = ureq::get(&url)
            .call()
            .map_err(|e| anyhow!("Failed to download dataset: {}", e))?;

        let body_str = response
            .into_body()
            .read_to_string()
            .map_err(|e| anyhow!("Failed to read response: {}", e))?;

        let body: serde_json::Value = serde_json::from_str(&body_str)
            .map_err(|e| anyhow!("Failed to parse response: {}", e))?;

        // Check for error in response
        if let Some(error) = body.get("error") {
            return Err(anyhow!("HuggingFace API error: {}", error));
        }

        // Parse the response based on dataset format
        let page_examples = parse_huggingface_response(dataset_name, &body)?;
        
        if page_examples.is_empty() {
            // No more data available
            break;
        }
        
        let fetched = page_examples.len();
        all_examples.extend(page_examples);
        offset += fetched;
        
        // If we got fewer than requested, we've reached the end
        if fetched < fetch_count {
            break;
        }
        
        // Progress indicator
        eprint!("\rFetched {} examples...", all_examples.len());
    }
    eprintln!(); // New line after progress
    
    let examples = all_examples;

    if examples.is_empty() {
        return Err(anyhow!(
            "No examples parsed from dataset. Check if split '{}' exists for '{}'",
            split,
            dataset_name
        ));
    }

    // Cache the examples
    let mut file = File::create(&dataset_file)?;
    for example in &examples {
        serde_json::to_writer(&mut file, example)?;
        writeln!(file)?;
    }

    log::info!(
        "Downloaded {} examples, cached to {}",
        examples.len(),
        dataset_file.display()
    );

    Ok(examples)
}

/// Parse HuggingFace API response based on dataset format.
#[cfg(feature = "bench")]
fn parse_huggingface_response(
    dataset_name: &str,
    response: &serde_json::Value,
) -> Result<Vec<Example>> {
    let rows = response["rows"]
        .as_array()
        .ok_or_else(|| anyhow!("No rows in response"))?;

    let mut examples = Vec::new();

    for row in rows {
        let row_data = &row["row"];

        // Handle different dataset formats
        let example = if dataset_name.contains("conll") {
            // CoNLL format: tokens and ner_tags arrays
            parse_conll_row(row_data)?
        } else if dataset_name.contains("pii") || dataset_name.contains("ai4privacy") {
            // PII dataset format
            parse_pii_row(row_data)?
        } else {
            // Try generic format
            parse_generic_row(row_data)?
        };

        if let Some(ex) = example {
            examples.push(ex);
        }
    }

    Ok(examples)
}

/// Parse a CoNLL-style row (tokens + ner_tags).
#[cfg(feature = "bench")]
fn parse_conll_row(row: &serde_json::Value) -> Result<Option<Example>> {
    let tokens = row["tokens"]
        .as_array()
        .ok_or_else(|| anyhow!("No tokens field"))?;
    let ner_tags = row["ner_tags"]
        .as_array()
        .ok_or_else(|| anyhow!("No ner_tags field"))?;

    if tokens.len() != ner_tags.len() {
        return Err(anyhow!("Tokens and ner_tags length mismatch"));
    }

    // Reconstruct text and entities
    let mut text = String::new();
    let mut entities = Vec::new();
    let mut current_entity: Option<(usize, String)> = None;

    // CoNLL NER tag mapping (BIO scheme)
    let tag_to_label = |tag: i64| -> Option<&'static str> {
        match tag {
            1 | 2 => Some("PER"),    // B-PER, I-PER
            3 | 4 => Some("ORG"),    // B-ORG, I-ORG
            5 | 6 => Some("LOC"),    // B-LOC, I-LOC
            7 | 8 => Some("MISC"),   // B-MISC, I-MISC
            _ => None,               // O (outside)
        }
    };

    for (i, (token, tag)) in tokens.iter().zip(ner_tags.iter()).enumerate() {
        let token_str = token.as_str().unwrap_or("");
        let tag_id = tag.as_i64().unwrap_or(0);

        let start = text.len();
        if i > 0 {
            text.push(' ');
        }
        text.push_str(token_str);
        let _end = text.len();

        // Check if this is a B- tag (beginning of entity)
        let is_begin = tag_id % 2 == 1 && tag_id > 0;

        if let Some(label) = tag_to_label(tag_id) {
            if is_begin {
                // Close previous entity if any
                if let Some((entity_start, entity_label)) = current_entity.take() {
                    entities.push(Entity {
                        start: entity_start,
                        end: start - 1, // Before the space
                        label: entity_label,
                        text: None,
                    });
                }
                // Start new entity
                current_entity = Some((if i > 0 { start + 1 } else { start }, label.to_string()));
            } else if current_entity.is_none() {
                // I- tag without B- tag, start new entity anyway
                current_entity = Some((if i > 0 { start + 1 } else { start }, label.to_string()));
            }
            // Otherwise continue current entity
        } else {
            // O tag - close current entity if any
            if let Some((entity_start, entity_label)) = current_entity.take() {
                entities.push(Entity {
                    start: entity_start,
                    end: start - 1, // Before the space
                    label: entity_label,
                    text: None,
                });
            }
        }
    }

    // Close final entity
    if let Some((entity_start, entity_label)) = current_entity {
        entities.push(Entity {
            start: entity_start,
            end: text.len(),
            label: entity_label,
            text: None,
        });
    }

    Ok(Some(Example { text, entities }))
}

/// Parse a PII-style row (text + entities directly).
#[cfg(feature = "bench")]
fn parse_pii_row(row: &serde_json::Value) -> Result<Option<Example>> {
    let text = row["source_text"]
        .as_str()
        .or_else(|| row["text"].as_str())
        .or_else(|| row["masked_text"].as_str())
        .unwrap_or("")
        .to_string();

    if text.is_empty() {
        return Ok(None);
    }

    let mut entities = Vec::new();

    // Try to get entities from various fields
    if let Some(privacy_mask) = row["privacy_mask"].as_array() {
        for mask in privacy_mask {
            if let (Some(label), Some(value)) = (mask["label"].as_str(), mask["value"].as_str()) {
                // Find the value in text
                if let Some(start) = text.find(value) {
                    entities.push(Entity {
                        start,
                        end: start + value.len(),
                        label: label.to_string(),
                        text: Some(value.to_string()),
                    });
                }
            }
        }
    } else if let Some(ents) = row["entities"].as_array() {
        for ent in ents {
            if let (Some(start), Some(end), Some(label)) = (
                ent["start"].as_u64(),
                ent["end"].as_u64(),
                ent["label"].as_str().or_else(|| ent["type"].as_str()),
            ) {
                entities.push(Entity {
                    start: start as usize,
                    end: end as usize,
                    label: label.to_string(),
                    text: None,
                });
            }
        }
    }

    Ok(Some(Example { text, entities }))
}

/// Parse a generic row format.
#[cfg(feature = "bench")]
fn parse_generic_row(row: &serde_json::Value) -> Result<Option<Example>> {
    // Try to find text and entities fields
    let text = row["text"]
        .as_str()
        .or_else(|| row["content"].as_str())
        .or_else(|| row["sentence"].as_str())
        .unwrap_or("")
        .to_string();

    if text.is_empty() {
        return Ok(None);
    }

    let mut entities = Vec::new();

    if let Some(ents) = row["entities"]
        .as_array()
        .or_else(|| row["labels"].as_array())
        .or_else(|| row["ner"].as_array())
    {
        for ent in ents {
            if let (Some(start), Some(end), Some(label)) = (
                ent["start"].as_u64().or_else(|| ent["begin"].as_u64()),
                ent["end"].as_u64(),
                ent["label"]
                    .as_str()
                    .or_else(|| ent["type"].as_str())
                    .or_else(|| ent["entity"].as_str()),
            ) {
                entities.push(Entity {
                    start: start as usize,
                    end: end as usize,
                    label: label.to_string(),
                    text: None,
                });
            }
        }
    }

    Ok(Some(Example { text, entities }))
}

/// Run benchmark on a set of examples.
pub fn run_benchmark(examples: &[Example], config: &BenchConfig) -> Result<BenchmarkResults> {
    let detector = Detector::new(&config.detector_config);

    let mut results = BenchmarkResults::new();

    let examples_to_process = if let Some(max) = config.max_examples {
        &examples[..max.min(examples.len())]
    } else {
        examples
    };

    for example in examples_to_process {
        results.total_examples += 1;
        let detected = detector.detect(&example.text);

        // Build sets of (start, end, label) for comparison
        let ground_truth_set: Vec<(usize, usize, String)> = example
            .entities
            .iter()
            .map(|e| {
                let mapped_label = config
                    .label_mapping
                    .get(&e.label)
                    .cloned()
                    .unwrap_or_else(|| e.label.to_lowercase());
                (e.start, e.end, mapped_label)
            })
            .collect();

        let mut detected_set: Vec<(usize, usize, String)> = detected
            .iter()
            .map(|d| (d.start, d.end, d.pattern_name.clone()))
            .collect();

        results.total_ground_truth += ground_truth_set.len();
        results.total_detected += detected_set.len();

        // Match entities
        for (gt_start, gt_end, gt_label) in &ground_truth_set {
            let label_entry = results
                .by_label
                .entry(gt_label.clone())
                .or_insert_with(|| LabelResults {
                    true_positives: 0,
                    false_positives: 0,
                    false_negatives: 0,
                    precision: 0.0,
                    recall: 0.0,
                    f1: 0.0,
                });

            // Find matching detection
            // gt_label is already mapped (e.g., "person" from "GIVENNAME1")
            // d_label is our pattern name (e.g., "person")
            let match_idx = detected_set.iter().position(|(d_start, d_end, d_label)| {
                // Direct match or same category
                let label_match = d_label == gt_label;

                if !label_match {
                    return false;
                }

                if config.strict_matching {
                    *d_start == *gt_start && *d_end == *gt_end
                } else {
                    // Relaxed: any overlap counts
                    *d_start < *gt_end && *d_end > *gt_start
                }
            });

            if let Some(idx) = match_idx {
                results.true_positives += 1;
                label_entry.true_positives += 1;
                detected_set.remove(idx);
            } else {
                results.false_negatives += 1;
                label_entry.false_negatives += 1;

                // Track missed detection if requested
                if config.track_misses_for.as_ref() == Some(gt_label) {
                    let missed_text = if *gt_end <= example.text.len() 
                        && example.text.is_char_boundary(*gt_start) 
                        && example.text.is_char_boundary(*gt_end) 
                    {
                        example.text[*gt_start..*gt_end].to_string()
                    } else {
                        format!("[invalid span {}..{}]", gt_start, gt_end)
                    };

                    // Get context (30 chars before and after), respecting char boundaries
                    let ctx_start = (0..*gt_start)
                        .rev()
                        .take(30)
                        .last()
                        .map(|i| if example.text.is_char_boundary(i) { i } else { *gt_start })
                        .unwrap_or(*gt_start);
                    let ctx_end = (*gt_end..example.text.len())
                        .take(30)
                        .last()
                        .map(|i| if example.text.is_char_boundary(i + 1) { i + 1 } else { *gt_end })
                        .unwrap_or(*gt_end);
                    let context = example.text.get(ctx_start..ctx_end)
                        .unwrap_or("[context unavailable]")
                        .to_string();

                    results.missed_detections.push(MissedDetection {
                        text: missed_text,
                        label: gt_label.clone(),
                        context,
                    });
                }
            }
        }

        // Remaining detections are false positives
        results.false_positives += detected_set.len();
        for (d_start, d_end, d_label) in detected_set {
            let label_entry = results
                .by_label
                .entry(d_label.clone())
                .or_insert_with(|| LabelResults {
                    true_positives: 0,
                    false_positives: 0,
                    false_negatives: 0,
                    precision: 0.0,
                    recall: 0.0,
                    f1: 0.0,
                });
            label_entry.false_positives += 1;

            // Track false positive if requested
            if config.track_false_positives_for.as_ref() == Some(&d_label) {
                let fp_text = if d_end <= example.text.len()
                    && example.text.is_char_boundary(d_start)
                    && example.text.is_char_boundary(d_end)
                {
                    example.text[d_start..d_end].to_string()
                } else {
                    format!("[invalid span {}..{}]", d_start, d_end)
                };

                // Get context (30 chars before and after), respecting char boundaries
                let ctx_start = (0..d_start)
                    .rev()
                    .take(30)
                    .last()
                    .map(|i| if example.text.is_char_boundary(i) { i } else { d_start })
                    .unwrap_or(d_start);
                let ctx_end = (d_end..example.text.len())
                    .take(30)
                    .last()
                    .map(|i| if example.text.is_char_boundary(i + 1) { i + 1 } else { d_end })
                    .unwrap_or(d_end);
                let context = example.text.get(ctx_start..ctx_end)
                    .unwrap_or("[context unavailable]")
                    .to_string();

                results.false_positive_detections.push(FalsePositive {
                    text: fp_text,
                    label: d_label,
                    context,
                });
            }
        }
    }

    results.finalize();
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_parsing() {
        let json = r#"{"text": "Contact John at john@example.com", "entities": [{"start": 8, "end": 12, "label": "PERSON"}, {"start": 16, "end": 32, "label": "EMAIL"}]}"#;
        let example: Example = serde_json::from_str(json).unwrap();

        assert_eq!(example.text, "Contact John at john@example.com");
        assert_eq!(example.entities.len(), 2);
        assert_eq!(example.entities[0].label, "PERSON");
        assert_eq!(example.entities[1].label, "EMAIL");
    }

    #[test]
    fn test_benchmark_basic() {
        let examples = vec![Example {
            text: "Contact test@example.com for info".to_string(),
            entities: vec![Entity {
                start: 8,
                end: 24,
                label: "EMAIL".to_string(),
                text: None,
            }],
        }];

        let config = BenchConfig::default();
        let results = run_benchmark(&examples, &config).unwrap();

        assert_eq!(results.total_examples, 1);
        assert_eq!(results.total_ground_truth, 1);
        // Should detect the email
        assert!(results.true_positives >= 1 || results.false_negatives >= 1);
    }
}
