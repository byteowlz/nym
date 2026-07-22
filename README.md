# nym

Fast, reversible PII anonymization CLI. Detect and anonymize personally identifiable information in text files with optional reversibility.

## Features

- **Pattern-based detection**: 50+ built-in regex patterns for emails, phone numbers, credit cards, SSNs, API keys, and more
- **NER integration**: AI-powered named entity recognition for names, organizations, and addresses (optional feature)
- **Reversible anonymization**: Store key files to restore original data later
- **Multiple formats**: Plain text, structured JSON, and in-place document redaction for docx/xlsx/pptx/ODF and PDF (see [docs/document-redaction.md](docs/document-redaction.md))
- **Streaming mode**: Process large files line-by-line with minimal memory usage
- **Flexible strategies**: Placeholder, masking, hashing, random, consistent, or fake replacements
- **Rulesets**: Predefined pattern groups for GDPR, HIPAA, programming, contact info, and more
- **Shell completions**: Generate completions for bash, zsh, fish, and more
- **XDG-compliant**: Respects standard config/data/state directories

## Installation

### From Source

```bash
# Default installation (streaming + progress bars)
cargo install --path .

# With NER support for AI-powered detection
cargo install --path . --features ner

# With all features (NER + bench)
cargo install --path . --all-features
```

### Platform-Specific NER Builds

```bash
# macOS Apple Silicon (CoreML acceleration)
cargo install --path . --features ner-coreml

# NVIDIA GPU (CUDA)
cargo install --path . --features ner-cuda

# Intel (OpenVINO)
cargo install --path . --features ner-openvino

# Windows (DirectML)
cargo install --path . --features ner-directml
```

### Interactive Install

```bash
just install
```

This detects your hardware and offers feature selection with sensible defaults.

## Quick Start

### Detect PII in a file

```bash
# Detect with default high-confidence patterns
nym detect input.txt

# Detect with all patterns
nym detect input.txt --min-confidence medium

# Output as JSON
nym detect input.txt --json

# Stream processing for large files
nym detect large.log --stream
```

### Anonymize PII

```bash
# Anonymize with default placeholder strategy
nym anon input.txt -o output.txt

# Anonymize with reversible key file
nym anon input.txt -o output.txt -k keys.jsonl

# Use fake but realistic replacements
nym anon input.txt --strategy fake

# Only anonymize contact information
nym anon input.txt --only-contact

# Use NER for names and addresses (requires ner feature)
ym anon input.txt --ner
```

### Redact documents in place

```bash
# Office formats: formatting preserved, fully reversible with a key file
nym anon report.docx -k keys.jsonl          # -> report.anon.docx
nym deanon report.anon.docx -k keys.jsonl   # restore

# PDF: TRUE redaction - text is removed from content streams and the output
# is verified to contain none of the redacted values (never a black box overlay)
nym anon contract.pdf                        # -> contract.anon.pdf
```

Covers hidden PII too: headers/footers, comments, speaker notes, spreadsheet
shared strings (formulas untouched), document metadata, PDF annotations.

```bash
# Scans & images: OCR-based redaction with re-OCR verification
# (engine: cargo install --path tools/nym-ocr, or tesseract)
nym anon scan.png                    # -> scan.anon.png
nym anon scanned-contract.pdf --ocr  # also redacts PII inside page images
```

### Restore Original Data

```bash
# De-anonymize using the key file
nym deanon output.txt -k keys.jsonl -o restored.txt
```

## Commands

### `nym detect [OPTIONS] [INPUT]`

Detect PII without modifying the text.

```bash
# JSON output with path info
nym detect data.json --format json --json

# Summary only
nym detect logs.txt --summary

# Use specific ruleset
nym detect file.txt -r gdpr

# Quick pattern shortcuts
nym detect file.txt --only-names      # Names, emails, SSNs
nym detect file.txt --only-keys       # API keys, tokens
nym detect file.txt --only-contact    # Email, phone, social
nym detect file.txt --only-financial  # Credit cards, IBANs
```

### `nym anon [OPTIONS] [INPUT]`

Anonymize PII in the input.

**Replacement strategies:**

- `placeholder`: Fixed placeholders like `<EMAIL>`, `<PHONE>`
- `mask`: Preserve length with X characters (`user@example.com` -> `XXXX@XXXXXXX.XXX`)
- `hash`: Deterministic hash-based replacement
- `random`: Random replacements
- `consistent`: Same input always gets same replacement (within session)
- `fake`: Realistic-looking fake data (default)

```bash
# JSON anonymization (preserves structure)
nym anon data.json --format json -o anonymized.json

# With session tag for tracking
nym anon input.txt -k keys.jsonl --tag "customer-data-2024"

# Exclude specific patterns
nym anon input.txt --exclude ssn,passport_us

# Only use specific patterns
nym anon input.txt --patterns email,phone_us
```

### `nym deanon [OPTIONS] [INPUT]`

Restore original data from anonymized text using a key file.

```bash
nym deanon anonymized.txt -k keys.jsonl -o restored.txt
```

### `nym patterns [ACTION]`

List and inspect available PII patterns.

```bash
# List all patterns
nym patterns list

# Show pattern details
nym patterns show email

# Test pattern against text
nym patterns test email "Contact us at support@example.com"

# List rulesets
nym patterns rulesets

# Show ruleset details
nym patterns ruleset gdpr
```

### `nym config [ACTION]`

Manage configuration.

```bash
# Show current config
nym config show

# Show config file path
nym config path

# Initialize config file
nym config init
```

### `nym sessions [ACTION]`

Search and manage anonymization sessions.

```bash
# List all key files
nym sessions list

# Search for session by ID
nym sessions search <session-id>
```

### `nym completions <SHELL>`

Generate shell completions.

```bash
nym completions bash > /path/to/completions/nym.bash
nym completions zsh > /path/to/completions/_nym
nym completions fish > /path/to/completions/nym.fish
```

## Configuration

Configuration file location (in order of priority):

1. `--config <path>` flag
2. `$XDG_CONFIG_HOME/nym/config.toml`
3. `~/.config/nym/config.toml`

### Example Configuration

```toml
"$schema" = "https://raw.githubusercontent.com/byteowlz/schemas/refs/heads/main/nym/nym.config.schema.json"

[detection]
enabled_patterns = ["email", "phone_us", "ssn", "credit_card"]
disabled_patterns = ["passport_us"]
min_confidence = "high"

[replacement]
strategy = "fake"
email_domain = "example.com"

[ner]
enabled = false
model = "onnx-community/gliner_multi-v2.1"
threshold = 0.5
labels = ["person", "organization", "street_address"]

[rulesets.my-custom]
description = "My custom ruleset"
enabled_patterns = ["email", "phone_us"]
min_confidence = "medium"
ner = "auto"
```

See `examples/config.toml` for a complete annotated example.

## Built-in Rulesets

| Ruleset      | Description                              |
|--------------|------------------------------------------|
| `minimal`    | Essential PII only (email, phone, SSN)   |
| `contact`    | Contact information                      |
| `identity`   | Personal identifiers                     |
| `financial`  | Credit cards, IBANs, bank accounts       |
| `programming`| API keys, tokens, credentials            |
| `gdpr`       | EU GDPR covered data                     |
| `hipaa`      | US healthcare data                       |
| `all`        | All available patterns                   |

## Built-in Patterns

**Personal identifiers**: SSNs, passport numbers, driver's licenses, birth dates, national IDs

**Contact**: Emails, phone numbers (US and international), social media handles

**Financial**: Credit cards, IBANs, bank accounts, SWIFT codes, crypto wallets

**Technical**: IPv4/IPv6 addresses, MAC addresses, UUIDs, URLs

**Credentials**: API keys, AWS keys, JWT tokens, private keys, passwords

**Vehicle**: VIN numbers, license plates

Run `nym patterns list` for the complete list with examples and confidence levels.

## NER Support

When built with the `ner` feature, nym can use AI-powered Named Entity Recognition via ONNX Runtime and GLiNER for detecting:

- Person names
- Organizations
- Street addresses
- Cities, countries
- Custom entity labels (zero-shot)

The default model (`onnx-community/gliner_multi-v2.1`) is Apache-2.0 licensed and supports 50+ languages.

### NER backends (GLiNER + OpenMed)

nym ships two NER backends that can run individually or together:

- **`gliner`** — GLiNER zero-shot span model; supply any labels you like.
- **`openmed`** — [OpenMed](https://github.com/maziyarpanahi/openmed) DeBERTa-v2 token-classification models fine-tuned for clinical/HIPAA PII, with a fixed 106-label taxonomy (names, dates of birth, medical record numbers, and more).
- **`both`** (default) — run both and merge results for best recall.

Select with `[ner] backend = "both" | "gliner" | "tokens"`. Token-classification models auto-download from the Hub (with a progress bar) — the default is nym's own small int8 model [`Wismut/nym-pii-multilingual-small/int8`](https://huggingface.co/Wismut/nym-pii-multilingual-small) (144 MB, 40 PII types, ~23 languages, OCR-noise-trained). Trade up for accuracy with [`Wismut/nym-pii-multilingual`](https://huggingface.co/Wismut/nym-pii-multilingual) (best, 1.2 GB) or `-small` fp32 (421 MB). Alternatives: `Wismut/openmed-onnx/{small,base,large}` (clinical), `nationaldesignstudio/rampart` (tiny), or a local dir. See [docs/ner-backends.md](docs/ner-backends.md) for the model-selection guide.

## Key Files

Key files store replacement mappings for reversibility:

```jsonl
{"version":"1.0","session":"nym-abc123","source":null,"tag":"customer-data","timestamp":"2024-01-15T10:30:00Z","generator":"nym 0.1.0"}
{"original":"john.doe@example.com","replacement":"sarah.smith@example.com","pattern":"email"}
{"original":"123-45-6789","replacement":"XXX-XX-XXXX","pattern":"ssn"}
```

Key files are append-only JSON Lines format for safe concurrent access.

## Development

```bash
# Run tests
just test

# Run with all features
just test-all

# Format code
just fmt

# Run clippy
just clippy

# Build release
just build-release

# Build with NER
just build-ner

# Example workflows
just example-detect
just example-anon
```

## Environment Variables

- `NYM__*`: Override any config value (e.g., `NYM__LOGGING__LEVEL=debug`)
- `NO_COLOR`: Disable ANSI colors
- `XDG_CONFIG_HOME`: Config directory
- `XDG_DATA_HOME`: Data directory
- `XDG_STATE_HOME`: State directory (for key files)

## License

MIT OR Apache-2.0
