# Privacy Filter Integration Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Integrate OpenAI's privacy-filter model (Apache 2.0, 1.5B params MoE) as a native Rust NER backend in nym, using ONNX Runtime for inference and a custom Rust Viterbi BIOES decoder.

**Architecture:** New `privacy-filter` feature flag that uses the existing `ort` + `tokenizers` crates (already in dependency tree via GLiNER). Downloads the q4 ONNX model (~809MB) from HuggingFace at runtime. Implements the constrained Viterbi decoder in pure Rust for BIOES span decoding from the 33-class token classifier output.

**Tech Stack:** `ort` 2.0.0-rc.9 (ONNX Runtime), `tokenizers` 0.21.4 (HF tokenizer), `hf-hub` (model download), `ndarray` (logit manipulation).

**Model Details:**
- Architecture: 8-layer GQA transformer with MoE FFN (128 experts, top-4 routing)
- `d_model=640`, 14 query heads, 2 KV heads, `head_dim=64`
- Banded attention (sliding_window=128)
- 200K vocab (GPT-OSS tokenizer)
- 33 output classes: O + 8 categories * 4 BIOES tags (B/I/E/S)
- Categories: account_number, private_address, private_date, private_email, private_person, private_phone, private_url, secret
- Viterbi calibration: 6 transition bias params (all 0.0 for default)

---

## Phase 1: Core Infrastructure

### Task 1: Add feature flag and dependencies to Cargo.toml

**Files:**
- Modify: `Cargo.toml`

**Step 1: Add the `privacy-filter` feature flag**

Add after the existing `ner-*` feature flags (around line 127):

```toml
# Privacy Filter (OpenAI privacy-filter model for PII detection)
privacy-filter = ["ort", "hf-hub", "libc", "tokenizers", "ndarray"]
privacy-filter-load-dynamic = ["privacy-filter", "ort/load-dynamic"]
privacy-filter-cuda = ["privacy-filter", "ort/cuda"]
privacy-filter-rocm = ["privacy-filter", "ort/rocm"]
privacy-filter-coreml = ["privacy-filter", "ort/coreml"]
privacy-filter-tensorrt = ["privacy-filter", "ort/tensorrt"]
```

Add `ndarray` to `[dependencies]`:

```toml
# Ndarray for logit manipulation (privacy-filter feature)
ndarray = { version = "0.16", optional = true }

# Tokenizers for privacy-filter model
tokenizers = { version = "0.21", optional = true }
```

Note: `ort`, `hf-hub`, and `libc` are already optional deps used by `ner`. The `privacy-filter` feature activates them too.

**Step 2: Verify Cargo.toml parses**

Run: `cd /Users/tommyfalkowski/byteowlz/nym && cargo check --features privacy-filter`
Expected: Compilation errors about unresolved imports (expected, we haven't written the code yet)

**Step 3: Commit**

```bash
git add Cargo.toml
git commit -m "feat: add privacy-filter feature flag and dependencies"
```

---

### Task 2: Create the privacy_filter module stub

**Files:**
- Create: `src/engine/privacy_filter.rs`
- Modify: `src/engine/mod.rs`

**Step 1: Create module stub**

Create `src/engine/privacy_filter.rs` with the module structure:

```rust
//! OpenAI Privacy Filter model integration.
//!
//! Implements the privacy-filter token classification model for PII detection.
//! Uses ONNX Runtime for inference and a native Rust Viterbi decoder for
//! BIOES span extraction.

/// The 33 BIOES label set for the privacy-filter model.
pub mod labels;

/// Constrained Viterbi decoder for BIOES span extraction.
pub mod viterbi;

/// ONNX model loading and inference.
#[cfg(feature = "privacy-filter")]
pub mod model;

/// Privacy Filter detector that integrates with nym's detection engine.
#[cfg(feature = "privacy-filter")]
pub mod detector;

pub use labels::PrivacyCategory;

#[cfg(feature = "privacy-filter")]
pub use detector::PrivacyFilterDetector;
```

**Step 2: Register module in mod.rs**

In `src/engine/mod.rs`, add after the `pub mod ner;` line:

```rust
#[cfg(feature = "privacy-filter")]
pub mod privacy_filter;
```

Add to the public exports:

```rust
#[cfg(feature = "privacy-filter")]
pub use privacy_filter::PrivacyFilterDetector;
```

**Step 3: Verify compilation**

Run: `cargo check --features privacy-filter`
Expected: Errors about missing submodules (expected)

**Step 4: Commit**

```bash
git add src/engine/privacy_filter.rs src/engine/mod.rs
git commit -m "feat: add privacy_filter module stub"
```

---

## Phase 2: BIOES Label System

### Task 3: Implement the BIOES label taxonomy

**Files:**
- Create: `src/engine/privacy_filter/labels.rs` (this file should be at `src/engine/privacy_filter/` — actually, let's keep it flat in `src/engine/privacy_filter.rs` for simplicity, with sub-modules inline)

Actually, let's restructure. Keep everything in `src/engine/privacy_filter.rs` as a single file to start, splitting into modules only if it gets too large.

**Revised approach:** Single file `src/engine/privacy_filter.rs` with all the logic.

**Files:**
- Rewrite: `src/engine/privacy_filter.rs`

**Step 1: Write failing tests for label taxonomy**

Create `src/engine/privacy_filter.rs`:

```rust
//! OpenAI Privacy Filter model integration.
//!
//! Implements the privacy-filter token classification model for PII detection.
//! Uses ONNX Runtime for inference and a native Rust Viterbi decoder for
//! BIOES span extraction.
//!
//! # Model Details
//!
//! - 1.5B params (50M active), MoE with 128 experts (top-4 routing)
//! - 8-layer GQA transformer, d_model=640
//! - 200K vocab (GPT-OSS tokenizer), 128K context window
//! - 33 output classes: O + 8 categories × 4 BIOES tags
//!
//! # Label Taxonomy (8 PII categories)
//!
//! 1. account_number
//! 2. private_address
//! 3. private_date
//! 4. private_email
//! 5. private_person
//! 6. private_phone
//! 7. private_url
//! 8. secret
//!
//! Each category has B (Begin), I (Inside), E (End), S (Single) variants,
//! plus the O (Outside/background) class, for 33 total classes.

/// The 8 PII categories detected by the privacy-filter model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrivacyCategory {
    AccountNumber,
    PrivateAddress,
    PrivateDate,
    PrivateEmail,
    PrivatePerson,
    PrivatePhone,
    PrivateUrl,
    Secret,
}

impl PrivacyCategory {
    /// All privacy categories in label-id order.
    pub const ALL: [Self; 8] = [
        Self::AccountNumber,
        Self::PrivateAddress,
        Self::PrivateDate,
        Self::PrivateEmail,
        Self::PrivatePerson,
        Self::PrivatePhone,
        Self::PrivateUrl,
        Self::Secret,
    ];

    /// Convert from the model's label string (e.g., "B-private_person").
    pub fn from_label(label: &str) -> Option<(BioesTag, Self)> {
        if label == "O" {
            return None;
        }

        let (tag_str, category_str) = label.split_once('-')?;
        let tag = match tag_str {
            "B" => BioesTag::Begin,
            "I" => BioesTag::Inside,
            "E" => BioesTag::End,
            "S" => BioesTag::Single,
            _ => return None,
        };

        let category = match category_str {
            "account_number" => Self::AccountNumber,
            "private_address" => Self::PrivateAddress,
            "private_date" => Self::PrivateDate,
            "private_email" => Self::PrivateEmail,
            "private_person" => Self::PrivatePerson,
            "private_phone" => Self::PrivatePhone,
            "private_url" => Self::PrivateUrl,
            "secret" => Self::Secret,
            _ => return None,
        };

        Some((tag, category))
    }

    /// Convert from label ID (0-32).
    pub fn from_id(id: usize) -> Option<(Option<BioesTag>, Self)> {
        if id == 0 {
            return None; // O (background)
        }
        let adjusted = id - 1;
        let tag = match adjusted % 4 {
            0 => BioesTag::Begin,
            1 => BioesTag::Inside,
            2 => BioesTag::End,
            3 => BioesTag::Single,
            _ => return None,
        };
        let cat_idx = adjusted / 4;
        let category = Self::ALL.get(cat_idx).copied()?;
        Some((Some(tag), category))
    }

    /// Convert to nym pattern name.
    pub fn to_pattern_name(self) -> &'static str {
        match self {
            Self::AccountNumber => "account_number",
            Self::PrivateAddress => "private_address",
            Self::PrivateDate => "private_date",
            Self::PrivateEmail => "private_email",
            Self::PrivatePerson => "private_person",
            Self::PrivatePhone => "private_phone",
            Self::PrivateUrl => "private_url",
            Self::Secret => "secret",
        }
    }

    /// Convert to nym PII category.
    pub fn to_nym_category(self) -> super::patterns::PiiCategory {
        match self {
            Self::AccountNumber => super::patterns::PiiCategory::Financial,
            Self::PrivateAddress => super::patterns::PiiCategory::Contact,
            Self::PrivateDate => super::patterns::PiiCategory::Identity,
            Self::PrivateEmail => super::patterns::PiiCategory::Contact,
            Self::PrivatePerson => super::patterns::PiiCategory::Identity,
            Self::PrivatePhone => super::patterns::PiiCategory::Contact,
            Self::PrivateUrl => super::patterns::PiiCategory::Other,
            Self::Secret => super::patterns::PiiCategory::Credentials,
        }
    }
}

impl std::fmt::Display for PrivacyCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.to_pattern_name())
    }
}

/// BIOES boundary tags for span labeling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BioesTag {
    /// Beginning of a multi-token span
    Begin,
    /// Inside a multi-token span
    Inside,
    /// End of a multi-token span
    End,
    /// Single-token span
    Single,
}

/// Number of output classes: 1 (O) + 8 categories * 4 BIOES tags = 33.
pub const NUM_CLASSES: usize = 33;

/// Label ID for the background (O) class.
pub const BACKGROUND_ID: usize = 0;

/// Build the id2label mapping matching the model's config.json.
pub fn id2label() -> [ &'static str; NUM_CLASSES] {
    let mut labels = ["O"; NUM_CLASSES];
    labels[0] = "O";
    let prefixes = ["B", "I", "E", "S"];
    let categories = [
        "account_number",
        "private_address",
        "private_date",
        "private_email",
        "private_person",
        "private_phone",
        "private_url",
        "secret",
    ];
    let mut idx = 1;
    for cat in &categories {
        for prefix in &prefixes {
            // We can't format at compile time in a const fn, so build at runtime
            // This is only used for debugging anyway
            idx += 1;
        }
    }
    // Actually, let's just hard-code the mapping to match config.json exactly:
    labels[0] = "O";
    labels[1] = "B-account_number";
    labels[2] = "I-account_number";
    labels[3] = "E-account_number";
    labels[4] = "S-account_number";
    labels[5] = "B-private_address";
    labels[6] = "I-private_address";
    labels[7] = "E-private_address";
    labels[8] = "S-private_address";
    labels[9] = "B-private_date";
    labels[10] = "I-private_date";
    labels[11] = "E-private_date";
    labels[12] = "S-private_date";
    labels[13] = "B-private_email";
    labels[14] = "I-private_email";
    labels[15] = "E-private_email";
    labels[16] = "S-private_email";
    labels[17] = "B-private_person";
    labels[18] = "I-private_person";
    labels[19] = "E-private_person";
    labels[20] = "S-private_person";
    labels[21] = "B-private_phone";
    labels[22] = "I-private_phone";
    labels[23] = "E-private_phone";
    labels[24] = "S-private_phone";
    labels[25] = "B-private_url";
    labels[26] = "I-private_url";
    labels[27] = "E-private_url";
    labels[28] = "S-private_url";
    labels[29] = "B-secret";
    labels[30] = "I-secret";
    labels[31] = "E-secret";
    labels[32] = "S-secret";
    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_roundtrip() {
        let labels = id2label();
        assert_eq!(labels[0], "O");
        assert_eq!(labels[17], "B-private_person");
        assert_eq!(labels[20], "S-private_person");
        assert_eq!(labels[29], "B-secret");
        assert_eq!(labels[32], "S-secret");

        // Verify from_label roundtrip
        for (id, label) in labels.iter().enumerate() {
            if *label == "O" {
                assert!(PrivacyCategory::from_label(label).is_none());
                assert!(PrivacyCategory::from_id(id).is_none());
            } else {
                let (tag, cat) = PrivacyCategory::from_label(label).unwrap();
                let (tag2, cat2) = PrivacyCategory::from_id(id).unwrap().unwrap();
                assert_eq!(tag, tag2);
                assert_eq!(cat, cat2);
            }
        }
    }

    #[test]
    fn test_num_classes() {
        assert_eq!(NUM_CLASSES, 33);
        let labels = id2label();
        assert_eq!(labels.len(), NUM_CLASSES);
    }

    #[test]
    fn test_category_to_pattern_name() {
        assert_eq!(PrivacyCategory::PrivatePerson.to_pattern_name(), "private_person");
        assert_eq!(PrivacyCategory::Secret.to_pattern_name(), "secret");
        assert_eq!(PrivacyCategory::PrivateEmail.to_pattern_name(), "private_email");
    }
}
```

Wait, the `id2label()` function signature is wrong — you can't return `&'static str` from a function that builds strings. Let me fix that. We'll use a const array approach:

```rust
/// Label ID to string mapping, matching config.json exactly.
pub const ID2LABEL: [&str; NUM_CLASSES] = [
    "O",
    "B-account_number", "I-account_number", "E-account_number", "S-account_number",
    "B-private_address", "I-private_address", "E-private_address", "S-private_address",
    "B-private_date", "I-private_date", "E-private_date", "S-private_date",
    "B-private_email", "I-private_email", "E-private_email", "S-private_email",
    "B-private_person", "I-private_person", "E-private_person", "S-private_person",
    "B-private_phone", "I-private_phone", "E-private_phone", "S-private_phone",
    "B-private_url", "I-private_url", "E-private_url", "S-private_url",
    "B-secret", "I-secret", "E-secret", "S-secret",
];
```

**Step 2: Verify tests pass**

Run: `cargo test --lib engine::privacy_filter --features privacy-filter`
Expected: 3 passing tests

**Step 3: Commit**

```bash
git add src/engine/privacy_filter.rs
git commit -m "feat: implement BIOES label taxonomy for privacy-filter"
```

---

### Task 4: Implement the constrained Viterbi decoder

**Files:**
- Modify: `src/engine/privacy_filter.rs` (add Viterbi decoder)

**Step 1: Write failing test for Viterbi decoder**

Add to the test module:

```rust
#[test]
fn test_viterbi_simple_span() {
    // Simulate logits for "My name is Alice Smith"
    // Tokens: ["My", "name", "is", "Alice", "Smith"]
    // Expected: O, O, O, B-private_person, E-private_person
    let logits = create_simple_logits();
    let result = viterbi_decode(&logits, &ViterbiConfig::default());
    assert_eq!(result[0], 0); // O
    assert_eq!(result[1], 0); // O
    assert_eq!(result[2], 0); // O
    assert_eq!(result[3], 17); // B-private_person
    assert_eq!(result[4], 19); // E-private_person
}

#[test]
fn test_viterbi_single_token_span() {
    // "My SSN is 123" — "123" should be S-account_number
    let logits = create_single_span_logits();
    let result = viterbi_decode(&logits, &ViterbiConfig::default());
    assert_eq!(result[3], 4); // S-account_number
}

#[test]
fn test_viterbi_all_background() {
    // "The weather is nice" — all O
    let logits = create_all_background_logits();
    let result = viterbi_decode(&logits, &ViterbiConfig::default());
    for &label in &result {
        assert_eq!(label, 0); // O
    }
}
```

**Step 2: Implement Viterbi decoder**

The decoder needs to enforce BIOES transition constraints. Key constraints:
- `O` can transition to `O`, `B-*`, or `S-*`
- `B-X` can transition to `I-X` or `E-X`
- `I-X` can transition to `I-X` or `E-X`
- `E-X` can transition to `O`, `B-Y`, or `S-Y`
- `S-X` can transition to `O`, `B-Y`, or `S-Y`

The Viterbi calibration adds 6 bias parameters:
- `transition_bias_background_stay`: bias for O → O
- `transition_bias_background_to_start`: bias for O → B or O → S
- `transition_bias_end_to_background`: bias for E → O or S → O
- `transition_bias_end_to_start`: bias for E → B, E → S, S → B, or S → S
- `transition_bias_inside_to_continue`: bias for B → I, I → I
- `transition_bias_inside_to_end`: bias for B → E, I → E

Add the Viterbi structs and decoder function.

**Step 3: Verify tests pass**

Run: `cargo test --lib engine::privacy_filter::tests::test_viterbi --features privacy-filter`

**Step 4: Commit**

```bash
git add src/engine/privacy_filter.rs
git commit -m "feat: implement constrained Viterbi BIOES decoder"
```

---

## Phase 3: Model Loading and Inference

### Task 5: Implement model download and ONNX loading

**Files:**
- Modify: `src/engine/privacy_filter.rs` (add model loading)

**Step 1: Implement `PrivacyFilterModel` struct**

This handles:
1. Downloading the ONNX model from HuggingFace via `hf-hub`
2. Loading the tokenizer from `tokenizer.json`
3. Creating the ONNX Runtime session
4. Running inference (input_ids + attention_mask → logits [B, T, 33])

The model download target: `openai/privacy-filter`, ONNX variant `model_q4.onnx` + `model_q4.onnx_data`.

Actually, the ONNX model uses external data files (`model_q4.onnx_data`). We need to download all the parts and make sure ort can find them. The `hf-hub` crate can handle this.

**Step 2: Write test for model loading**

Create an integration test (behind `#[ignore]` since it downloads a large model):

```rust
#[test]
#[ignore] // Requires model download (~809MB)
fn test_model_load_and_inference() {
    let model = PrivacyFilterDetector::new(None, None).unwrap();
    let result = model.detect("My name is Alice Smith").unwrap();
    assert!(!result.is_empty());
}
```

**Step 3: Commit**

```bash
git add src/engine/privacy_filter.rs
git commit -m "feat: implement ONNX model loading and inference"
```

---

### Task 6: Implement the PrivacyFilterDetector

**Files:**
- Modify: `src/engine/privacy_filter.rs` (add detector)
- Modify: `src/engine/detector.rs` (integrate)

**Step 1: Implement PrivacyFilterDetector**

This is the main struct that:
1. Tokenizes input text
2. Runs ONNX inference
3. Applies Viterbi decoding to get BIOES labels
4. Extracts spans and maps them to nym's PiiMatch format
5. Maps byte offsets from tokenizer back to original text

The key challenge is mapping token offsets back to character offsets. The HuggingFace tokenizer returns byte-level offsets, which we can convert.

**Step 2: Integrate with Detector struct**

In `src/engine/detector.rs`, add a `privacy_filter_detector` field behind `#[cfg(feature = "privacy-filter")]`, similar to the existing `ner_detector`.

Add to `DetectorConfig`:
```rust
#[cfg(feature = "privacy-filter")]
pub privacy_filter_enabled: bool,
```

**Step 3: Verify compilation**

Run: `cargo check --features privacy-filter`

**Step 4: Commit**

```bash
git add src/engine/privacy_filter.rs src/engine/detector.rs src/engine/mod.rs
git commit -m "feat: implement PrivacyFilterDetector and integrate with Detector"
```

---

## Phase 4: CLI Integration

### Task 7: Add `--privacy-filter` CLI flag

**Files:**
- Modify: `src/main.rs`

**Step 1: Add CLI flags**

Add to `AnonCommand` and `DetectCommand`:

```rust
/// Use OpenAI Privacy Filter model for PII detection (requires 'privacy-filter' feature)
#[arg(long)]
privacy_filter: bool,
```

**Step 2: Wire up in command handlers**

In the detect/anon handlers, when `privacy_filter` is true, enable the privacy filter detector.

**Step 3: Verify compilation**

Run: `cargo check --features privacy-filter`

**Step 4: Commit**

```bash
git add src/main.rs
git commit -m "feat: add --privacy-filter CLI flag"
```

---

### Task 8: Add config support for privacy-filter

**Files:**
- Modify: `src/config.rs`

**Step 1: Add PrivacyFilterConfig**

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PrivacyFilterConfig {
    pub enabled: bool,
    pub model: String,
    pub cache_dir: Option<String>,
    pub operating_point: String,
}

impl Default for PrivacyFilterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: "openai/privacy-filter".to_string(),
            cache_dir: None,
            operating_point: "default".to_string(),
        }
    }
}
```

**Step 2: Add to Config struct**

Add `pub privacy_filter: PrivacyFilterConfig` to the `Config` struct.

**Step 3: Verify compilation**

Run: `cargo check --features privacy-filter`

**Step 4: Commit**

```bash
git add src/config.rs
git commit -m "feat: add privacy-filter config support"
```

---

## Phase 5: Integration Tests & Polish

### Task 9: End-to-end integration test

**Files:**
- Create: `tests/privacy_filter.rs`

**Step 1: Write integration test**

```rust
#[cfg(feature = "privacy-filter")]
#[test]
#[ignore] // Requires model download
fn test_detect_person() {
    let output = std::process::Command::new("cargo")
        .args(["run", "--features", "privacy-filter", "--", "detect", "--privacy-filter", "--json"])
        .stdin(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    // Write "My name is Alice Smith" to stdin
    // Assert output contains "private_person" match
}
```

**Step 2: Commit**

```bash
git add tests/privacy_filter.rs
git commit -m "test: add privacy-filter integration tests"
```

---

### Task 10: Update README and justfile

**Files:**
- Modify: `README.md`
- Modify: `justfile`

Add documentation for the `privacy-filter` feature and build commands.

---

## Key Implementation Notes

### ONNX Model File Structure

The q4 model on HuggingFace consists of:
- `onnx/model_q4.onnx` (160 KB — graph definition)
- `onnx/model_q4.onnx_data` (917 MB — quantized weights)

Both files must be in the same directory for ONNX Runtime to find the external data.

### Tokenizer

The tokenizer uses HuggingFace's `tokenizers` library format (`tokenizer.json`, 27.9MB). It's a BPE tokenizer with 200K vocab, compatible with the `tokenizers` Rust crate.

### Viterbi Decoder

The decoder enforces BIOES transition constraints using dynamic programming:
1. Build a transition score matrix (33 × 33) with -inf for invalid transitions
2. Add calibration biases to valid transitions
3. Run standard Viterbi: for each token, accumulate max score path
4. Backtrack to get the best label sequence

### macOS Crash Workaround

Reuse the same ONNX Runtime crash workaround from the GLiNER module (ManuallyDrop + _exit).

### Model Size

The q4 model is ~809MB. Consider:
- Showing a download progress bar
- Caching in `~/.cache/huggingface/hub/`
- Allowing users to pre-download with a `nym models download` command
