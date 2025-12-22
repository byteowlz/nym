# Nym - PII Anonymization CLI: Comprehensive Implementation Plan

## Research Summary (December 2025 SOTA)

### Existing Tools Analyzed

| Tool | Language | Approach | Strengths | Weaknesses |
|------|----------|----------|-----------|------------|
| **redacter-rs** | Rust | External DLP APIs (GCP, AWS, Presidio) | Multi-format, cloud DLP integration | Requires external services, not standalone |
| **biip** | Rust | Regex-based, env-aware | Fast, offline, simple | Limited PII types, no reversal capability |
| **Microsoft Presidio** | Python | NLP + regex hybrid | Accurate, customizable | Python, heavy dependencies |
| **OpenPipe pii-redact** | Python | ML-based (fine-tuned models) | SOTA accuracy for unstructured text | Heavy, requires models |
| **data_privacy** (Microsoft Oxidizer) | Rust | Type-safe wrappers + redaction engine | Compile-time safety, taxonomy system | No detection, not for arbitrary text |

### Evaluated but Not Used: `data_privacy` Crate

The `data_privacy` crate from Microsoft's Oxidizer project was evaluated but **not selected** for this project. Here's the rationale:

**What `data_privacy` does:**
- Provides type-safe wrappers (`#[classified]` macro) for sensitive data
- Taxonomy system for categorizing PII types (e.g., `CustomerIdentifier`)
- Redaction engine with strategies (hash, asterisks, erase)
- Compile-time enforcement preventing accidental exposure via `Debug`/`Display`

**Why it doesn't fit our use case:**

| Requirement | `data_privacy` | Our Approach |
|-------------|----------------|--------------|
| Detect PII in arbitrary text | No - requires pre-wrapped types | Regex pattern matching |
| Process external files (JSON, TOML) | No - for in-memory structs only | Format-aware handlers |
| Reversible anonymization | No - hash/mask only | JSONL key file storage |
| CLI tool for end users | No - library for Rust apps | clap-based CLI |
| Streaming large files | No | Line-by-line processing |

**When `data_privacy` IS appropriate:**
- Building Rust services where you control the data model
- Wrapping known sensitive fields at compile time
- Preventing accidental PII leakage in logs/telemetry
- Enterprise applications with formal data taxonomies

**Conclusion:** `data_privacy` solves a complementary but different problem (compile-time PII safety for application developers) whereas `nym` solves runtime PII detection and anonymization in arbitrary external data.

### Key Insights

1. **Regex vs ML Trade-off**: Regex is fast but limited for unstructured text. ML is accurate but slow and heavy.
2. **Hybrid Approach**: Best tools combine regex for structured patterns (SSN, email, phone) with optional ML for names/addresses.
3. **Reversal Capability**: No existing Rust tool supports reversible PII replacement with key storage.
4. **Format-Aware Processing**: JSON/TOML need structure-preserving replacement; plain text is simpler.

---

## Design Philosophy

**Name**: `nym` (from "pseudonym" - a fictitious name used to conceal identity)

**Core Principles**:
1. **Offline-first**: No external API dependencies for basic operation
2. **Blazingly fast**: Use `regex` crate with `RegexSet` for parallel pattern matching
3. **Reversible**: Optional key-value mapping storage for de-anonymization
4. **Format-aware**: Preserve structure in JSON, TOML, YAML while anonymizing values
5. **Streaming**: Process arbitrarily large files without loading entirely into memory
6. **Configurable**: User-defined patterns, sensitivity levels, replacement strategies

---

## Architecture Overview

```
                    +-----------------+
                    |   CLI Interface |
                    |  (clap + stdin) |
                    +-----------------+
                            |
                    +-----------------+
                    |  Format Router  |
                    | (detect/parse)  |
                    +-----------------+
                            |
        +-------------------+-------------------+
        |                   |                   |
   +--------+          +--------+          +--------+
   |  Text  |          |  JSON  |          |  TOML  |
   | Handler|          | Handler|          | Handler|
   +--------+          +--------+          +--------+
        |                   |                   |
        +-------------------+-------------------+
                            |
                    +-----------------+
                    |   PII Engine    |
                    | (RegexSet +     |
                    |  replacers)     |
                    +-----------------+
                            |
                    +-----------------+
                    |  Key Store      |
                    | (optional JSONL)|
                    +-----------------+
                            |
                    +-----------------+
                    |    Output       |
                    +-----------------+
```

---

## PII Detection Patterns (Built-in)

### Tier 1: High Confidence (Structured Patterns)

| PII Type | Regex Pattern | Replacement Strategy |
|----------|---------------|---------------------|
| Email | `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}` | `user_<hash>@example.com` |
| Phone (US) | `\+?1?[-.\s]?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}` | `555-XXX-XXXX` |
| Phone (Intl) | `\+\d{1,3}[-.\s]?\d{1,14}` | `+XX-XXX-XXXX` |
| SSN | `\d{3}-\d{2}-\d{4}` | `XXX-XX-XXXX` |
| Credit Card | `\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}` | `XXXX-XXXX-XXXX-XXXX` |
| IPv4 | `\b(?:\d{1,3}\.){3}\d{1,3}\b` | `X.X.X.X` |
| IPv6 | `([a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}` | `XXXX:...:XXXX` |
| MAC Address | `([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}` | `XX:XX:XX:XX:XX:XX` |
| UUID | `[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}` | `<uuid>` |

### Tier 2: Medium Confidence

| PII Type | Regex Pattern | Notes |
|----------|---------------|-------|
| Date of Birth | `\d{1,2}[/-]\d{1,2}[/-]\d{2,4}` | Context-sensitive |
| API Keys | Various provider patterns | AWS, OpenAI, Stripe, etc. |
| JWT | `eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*` | Base64 JWT format |
| Passport | `[A-Z]{1,2}\d{6,9}` | Country-specific |
| IBAN | `[A-Z]{2}\d{2}[A-Z0-9]{11,30}` | International bank accounts |

### Tier 3: Optional/Custom

- Street addresses (configurable, high false positive rate)
- Names (requires word lists or ML - optional feature)
- Custom patterns from config

---

## CLI Design

### Commands

```bash
# Basic anonymization (stdin/stdout)
nym anon < input.txt > output.txt
nym anon input.txt -o output.txt

# With reversible key storage
nym anon input.json -o output.json --key-file keys.jsonl

# Reverse anonymization
nym deanon output.json -o restored.json --key-file keys.jsonl

# Detect PII without replacing
nym detect input.txt --format json

# Interactive mode (like biip)
nym anon  # reads stdin, outputs to stdout

# Validate key file
nym keys validate keys.jsonl

# Show built-in patterns
nym patterns list
nym patterns show email
```

### Subcommand Structure

```
nym
├── anon       Anonymize PII in input
├── deanon     Restore PII from key file
├── detect     Detect and report PII without modification
├── patterns   Manage detection patterns
│   ├── list   List all available patterns
│   ├── show   Show pattern details
│   └── test   Test pattern against input
├── keys       Key file management
│   ├── validate  Validate key file integrity
│   ├── merge     Merge multiple key files
│   └── stats     Show key file statistics
├── init       Create config directories and defaults
├── config     Configuration management
└── completions  Generate shell completions
```

### Global Options (inherited from template)

```
-c, --config <PATH>     Override config file
-q, --quiet             Reduce output to errors only
-v, --verbose           Increase verbosity (stackable)
--json                  Output as JSON
--yaml                  Output as YAML
--dry-run               Show what would be done
--no-color              Disable colors
```

### Anonymization-Specific Options

```
nym anon [OPTIONS] [INPUT]

Arguments:
  [INPUT]  Input file (default: stdin)

Options:
  -o, --output <FILE>       Output file (default: stdout)
  -k, --key-file <FILE>     Store replacement mappings for reversal
  -f, --format <FORMAT>     Force input format [auto|text|json|toml|yaml|csv|md]
  --patterns <PATTERNS>     Comma-separated pattern names to use [default: all]
  --exclude <PATTERNS>      Patterns to exclude
  --replacement <STRATEGY>  Replacement strategy [random|consistent|hash|mask]
  --seed <SEED>             Seed for deterministic random replacements
  --parallel <N>            Parallel processing threads
  --in-place                Modify input file in place (requires backup or --force)
```

---

## Key File Format (JSONL)

```jsonl
{"version":"1","file":"input.json","created":"2024-12-19T10:00:00Z","checksum":"sha256:..."}
{"original":"john.doe@example.com","replacement":"user_a7f3@example.com","type":"email","locations":[{"line":5,"col":12}]}
{"original":"555-123-4567","replacement":"555-XXX-0001","type":"phone","locations":[{"line":10,"col":25}]}
```

Benefits:
- Append-only, streamable
- Each line is self-contained
- Can process huge files
- Easy to merge/split

---

## Configuration Schema

### `~/.config/nym/config.toml`

```toml
# Configuration for nym PII anonymization tool

profile = "default"

[detection]
# Built-in patterns to enable (default: all)
enabled_patterns = ["email", "phone", "ssn", "credit_card", "ipv4", "uuid"]

# Patterns to disable
disabled_patterns = []

# Minimum confidence level: "high", "medium", "low"
min_confidence = "high"

[replacement]
# Strategy: "random", "consistent", "hash", "mask"
# - random: Generate random replacements each time
# - consistent: Same input always gets same replacement (within session)
# - hash: Deterministic hash-based replacement
# - mask: Simple masking (XXX-XX-XXXX)
strategy = "consistent"

# Seed for deterministic replacements (optional)
# seed = 42

# Domain for email replacements
email_domain = "example.com"

[keys]
# Default key file location
default_key_file = "$XDG_STATE_HOME/nym/keys.jsonl"

# Auto-generate key file if not specified
auto_generate = false

# Key file retention (days, 0 = forever)
retention_days = 30

[formats]
# Format-specific settings
[formats.json]
preserve_structure = true
anonymize_keys = false  # Only anonymize values, not JSON keys

[formats.csv]
has_headers = true
delimiter = ","

[patterns.custom]
# Custom patterns defined by user
# [[patterns.custom.rules]]
# name = "employee_id"
# regex = "EMP-\\d{6}"
# replacement = "EMP-XXXXXX"
# confidence = "high"

[logging]
level = "info"

[runtime]
parallelism = 0  # 0 = auto-detect
buffer_size = 65536  # 64KB read buffer

[paths]
data_dir = "$XDG_DATA_HOME/nym"
state_dir = "$XDG_STATE_HOME/nym"
```

---

## Crate Dependencies

```toml
[dependencies]
# Core
anyhow = "1.0"
thiserror = "1.0"

# CLI
clap = { version = "4.5", features = ["derive", "env", "wrap_help"] }
clap_complete = "4.5"

# Configuration
config = { version = "0.15", features = ["toml"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
toml = "0.8"
serde_yaml = "0.9"

# Regex (core functionality)
regex = "1.10"
once_cell = "1.19"  # For lazy static regex compilation

# Random generation
rand = "0.8"
rand_chacha = "0.3"  # For seedable RNG

# Hashing
sha2 = "0.10"
base64 = "0.22"

# File handling
shellexpand = "3.1"
dirs = "5.0"
memmap2 = "0.9"  # For large file memory mapping

# Logging
log = "0.4"
env_logger = "0.11"

# Time
chrono = { version = "0.4", features = ["serde"] }

# CSV handling
csv = "1.3"

# Optional: Progress bars
indicatif = { version = "0.17", optional = true }

[features]
default = ["progress"]
progress = ["indicatif"]
```

---

## Module Structure

```
src/
├── main.rs              # Entry point, CLI parsing
├── lib.rs               # Library interface for programmatic use
├── cli/
│   ├── mod.rs           # CLI module
│   ├── anon.rs          # Anonymization command
│   ├── deanon.rs        # De-anonymization command
│   ├── detect.rs        # Detection command
│   ├── patterns.rs      # Pattern management commands
│   └── keys.rs          # Key file commands
├── config/
│   ├── mod.rs           # Configuration loading
│   └── schema.rs        # Config structs
├── engine/
│   ├── mod.rs           # PII engine
│   ├── patterns.rs      # Pattern definitions
│   ├── detector.rs      # Detection logic
│   └── replacer.rs      # Replacement strategies
├── formats/
│   ├── mod.rs           # Format router
│   ├── text.rs          # Plain text handler
│   ├── json.rs          # JSON handler
│   ├── toml.rs          # TOML handler
│   ├── yaml.rs          # YAML handler
│   ├── csv.rs           # CSV handler
│   └── markdown.rs      # Markdown handler
├── keys/
│   ├── mod.rs           # Key storage
│   ├── store.rs         # JSONL key store
│   └── mapping.rs       # Original <-> replacement mapping
├── output/
│   ├── mod.rs           # Output formatting
│   └── report.rs        # Detection reports
└── utils/
    ├── mod.rs           # Utilities
    ├── hash.rs          # Hashing utilities
    └── random.rs        # Random generation
```

---

## Implementation Phases

### Phase 1: Core Foundation (MVP)
1. Refactor existing template structure
2. Implement PII engine with RegexSet
3. Add basic text anonymization (stdin/stdout)
4. Implement `anon` command with file input/output
5. Add `detect` command for PII detection

### Phase 2: Reversibility
1. Implement key file format (JSONL)
2. Add key storage during anonymization
3. Implement `deanon` command
4. Add key file management commands

### Phase 3: Format Support
1. JSON-aware anonymization (preserve structure)
2. TOML-aware anonymization
3. YAML-aware anonymization
4. CSV handling with header awareness
5. Markdown handling

### Phase 4: Configuration & Polish
1. Full configuration system
2. Custom pattern support
3. Performance optimization (parallel processing)
4. Progress indicators
5. Comprehensive error handling

### Phase 5: Advanced Features
1. Large file support (memory mapping)
2. Streaming mode for pipes
3. API key detection (provider-specific)
4. Interactive mode
5. Shell completions

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Throughput (text) | > 100 MB/s |
| Memory (streaming) | < 10 MB for any file size |
| Startup time | < 50ms |
| Pattern compilation | Once at startup (lazy_static) |

### Optimization Strategies

1. **RegexSet**: Compile all patterns into a single RegexSet for parallel matching
2. **Memory mapping**: Use `memmap2` for large files
3. **Streaming**: Process line-by-line for text, don't load entire file
4. **Parallel processing**: Use `rayon` for multi-file operations
5. **Lazy compilation**: Compile regex only when needed

---

## Testing Strategy

1. **Unit tests**: Each pattern, each replacer, each format handler
2. **Integration tests**: Full CLI invocations
3. **Fuzz testing**: Random input to find edge cases
4. **Benchmark tests**: Ensure performance targets are met
5. **Golden tests**: Known input -> expected output

### Test Data

- Sample files with various PII types
- Edge cases (Unicode, mixed encodings)
- Large files (1GB+) for performance testing
- Format-specific test cases (malformed JSON, etc.)

---

## Security Considerations

1. **Key file protection**: Warn if key file has permissive permissions
2. **Secure deletion**: Option to securely delete original files
3. **No logging of PII**: Never log actual PII values
4. **Memory safety**: Clear sensitive data from memory when done
5. **Checksum validation**: Verify file integrity for de-anonymization

---

## Future Enhancements (Post-MVP)

1. **ML-based detection**: Optional integration with local models for names/addresses
2. **Plugin system**: User-defined detectors/replacers
3. **Watch mode**: Monitor directories for new files
4. **Git integration**: Pre-commit hook for PII detection
5. **IDE integration**: LSP server for real-time PII highlighting
6. **Localization**: Support for non-US PII formats (EU, Asia, etc.)

---

## Appendix: Crate Evaluation Notes

### Crates Considered

| Crate | Version | Decision | Notes |
|-------|---------|----------|-------|
| `regex` | 1.10+ | **USE** | Core pattern matching, RegexSet for parallel matching |
| `data_privacy` | 0.10 | **SKIP** | Wrong problem space (see above) |
| `biip` | 0.9 | **REFERENCE** | Good regex patterns to learn from, but no reversal |
| `once_cell` | 1.19 | **USE** | Lazy static regex compilation |
| `memmap2` | 0.9 | **USE** | Large file handling |

### Why Not Use Existing PII Tools as Dependencies?

1. **redacter-rs**: Requires external DLP APIs (GCP, AWS) - we want offline-first
2. **biip**: GPL-3.0 license, no library interface, no reversal capability
3. **presidio-rs**: Doesn't exist - Presidio is Python-only
4. **data_privacy**: Compile-time wrappers, not runtime detection

### Regex Crate Selection

The Rust `regex` crate was chosen over alternatives because:
- **RegexSet**: Can match multiple patterns in a single pass
- **No backtracking**: Guaranteed O(n) time complexity, no ReDoS vulnerabilities
- **Unicode support**: Full Unicode character class support
- **Battle-tested**: Used by ripgrep, widely audited

---

## Appendix: Validation Datasets

### Recommended Datasets (FOSS, MIT/Apache-2.0 compatible)

| Dataset/Tool | License | Description | Use Case |
|--------------|---------|-------------|----------|
| **fake-rs** (Rust crate) | MIT/Apache-2.0 | Rust library for generating fake PII data | Generate test fixtures programmatically |
| **Microsoft Presidio test data** | MIT | Test cases from Presidio's test suite | Reference patterns and edge cases |
| **Synthea** | Apache-2.0 | Synthetic patient/healthcare data generator | Healthcare PII patterns |
| **pseudopeople** | BSD-3 | Synthetic US population data (names, SSNs, addresses) | Entity resolution testing |

### Strategy: Generate Our Own Test Fixtures

Since most PII datasets have restrictive licenses (to prevent misuse), the recommended approach is:

1. **Use `fake` crate** (MIT/Apache-2.0) to generate synthetic PII in Rust tests:
   ```rust
   use fake::{Fake, faker::internet::en::SafeEmail, faker::phone_number::en::PhoneNumber};
   
   let email: String = SafeEmail().fake();
   let phone: String = PhoneNumber().fake();
   ```

2. **Create golden test files** with synthetic data covering:
   - All supported PII types (email, phone, SSN, credit card, etc.)
   - Edge cases (Unicode emails, international phone formats)
   - Format variations (JSON, TOML, YAML, CSV, plain text)
   - False positive scenarios (dates that look like SSNs, etc.)

3. **Reference Presidio's test patterns** (MIT licensed):
   - https://github.com/microsoft/presidio/tree/main/presidio-analyzer/tests
   - Contains regex patterns and test strings for validation

### Test Data Categories

```
test-fixtures/
├── positive/           # Should be detected
│   ├── emails.txt
│   ├── phones.txt
│   ├── ssn.txt
│   ├── credit_cards.txt
│   ├── mixed.json
│   └── mixed.toml
├── negative/           # Should NOT be detected (false positive tests)
│   ├── dates.txt       # Dates that look like SSNs
│   ├── ids.txt         # Product IDs, order numbers
│   └── numbers.txt     # Random number sequences
├── formats/            # Format-specific tests
│   ├── nested.json
│   ├── config.toml
│   ├── data.yaml
│   └── records.csv
└── edge_cases/         # Unicode, malformed, large files
    ├── unicode_emails.txt
    ├── intl_phones.txt
    └── large_file.txt
```

### Datasets NOT Recommended (License Issues)

| Dataset | License | Issue |
|---------|---------|-------|
| BigCode PII Dataset | Restrictive | Requires agreement, no redistribution |
| Kaggle PII datasets | Various | Often no commercial use |
| Real leaked data | N/A | Ethical and legal issues |

### Build-time Test Data Generation

Add to `Cargo.toml` for dev dependencies:
```toml
[dev-dependencies]
fake = { version = "4", features = ["derive"] }
```

This allows generating fresh test data for each test run, ensuring:
- No stale test data
- Randomized edge cases
- No licensing concerns (we generate, not redistribute)
