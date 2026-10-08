# Corpus vocabulary workflow

`nym terms` discovers recurring literals, lets a human review contexts, and exports explicitly approved sensitive terms. It works without NER or a network service; it is not specific to any agent harness. Discovery does not certify a corpus as clean.

## Use

Keep inputs and generated artifacts in a private directory outside git:

```bash
nym terms discover notes.txt logs.jsonl --output discovery.json
nym terms review discovery.json --html --output review.html
# Open the private HTML file, review suggestions, download nym-terms-review.json.
nym terms export discovery.json --review nym-terms-review.json --output sensitive.txt
nym anon notes.txt --sensitive-terms-file sensitive.txt --key-file keys.json --output clean.txt
```

The HTML is self-contained and phone-sized, with no uploads or external assets. It is not a hosted/authenticated service. If sharing it across devices, use private authenticated hosting or file transfer; the HTML contains private text. Browser resume is best-effort: download decisions before closing. Reopen with `--resume nym-terms-review.json` to embed saved decisions. Actual iOS browser/download behavior must be checked on the target device.

Machine review also works without HTML:

```bash
nym terms review discovery.json --output review.json --decision CANDIDATE_ID=sensitive
nym terms review discovery.json --resume review.json --output revised.json --decision OTHER_ID=contextual
nym --json terms export discovery.json --review revised.json --output sensitive.txt --dry-run
```

IDs appear in the discovery artifact. Decisions are `sensitive`, `contextual`, `dismiss`, or `unsure`. Only `sensitive` exports; pending, dismissed and uncertain terms never become implicit approvals or benign exemptions. CLI summaries contain counts, not corpus values. JSON/YAML summary output and shell completions use the normal NYM flags.

## Recursive directories

Pass a directory directly; NYM resolves the file set internally, so shell argument limits do not apply:

```bash
nym terms discover ~/logs --recursive --extension jsonl,ndjson \
  --include '**.text' --include '**.thinking' --phrase-words 1 \
  --max-distinct 200000 --output discovery.json
```

`--recursive` (`-r`) is required for directory roots. Multiple roots/files can be mixed. The entire canonical path set is resolved before extraction, sorted and deduplicated across overlapping roots/explicit files. File bytes are not snapshotted: freeze live sources separately when reproducibility is required. Hidden entries are included; no gitignore filters apply. This produces **one combined discovery artifact**, not one deck per subdirectory.

Directory files default to extensions `txt,text,log,md,json,jsonl,ndjson` (case-insensitive). Repeat `--extension` or use commas to replace that directory filter; `.JSONL` is accepted. Explicit named files retain normal format resolution regardless of the directory filter. `--format` overrides parsing, not extension selection: use `--extension jsonl,ndjson --format jsonl` for an explicitly JSONL-only tree.

Symlink entries are never followed and are counted as skipped; explicit symlink inputs are rejected (use the real path). Unsupported extensions are counted as skipped, not evidence of safety. Nonregular entries, inaccessible directories/selected files, malformed selected input, missing roots and trees without selected files fail operationally. Named output/background files are automatically excluded from traversal, including pre-existing output under `--force`; use repeatable `--exclude-file PATH` for other exact artifact paths. Artifacts explicitly supplied as inputs are errors. Keep generated files outside input trees where possible.

Traversal has a `--max-files` budget (default/hard ceiling 10,000 unique selected files), 100,000 inspected directory entries and 64 nested directories. Failed traversal or extraction never publishes a partial deck or overwrites a destination. Successful JSON/YAML and human summaries report selected files and skipped extension/symlink/artifact counts without paths or corpus values. Discovery vocabulary/unit/source-pair limits below remain separate: recursive input does **not** promise arbitrary whole-history scalability.

## Inputs and statistics

Supports text, JSON and JSONL/NDJSON, multiple named files, recursive directories or stdin. Binary documents must first be extracted to supported text; this command does not yet reuse PDF/office/OCR extraction. `--format` overrides conservative format resolution. JSON string values are decoded, not raw escape sequences. `--include`/`--exclude` use the existing NYM selectors, applied within each JSONL record. Text units are physical lines; offsets are UTF-8 bytes within the original decoded unit, not the complete input file. JSONL examples retain physical `record[N]` locations, including blank-line numbering.

Candidates include literal words/identifiers, one-to-three-word phrases, whole hosts, URLs, emails and absolute paths, plus components. Two-to-120-codepoint literals are eligible; phrases never cross lines or non-space punctuation. Numeric and dictionary words are not automatically excluded. No Unicode normalization or case folding is applied to stored literals.

Counts include every occurrence, including exact copied output. Ranking without background prioritizes distinct exact text units, then distinct canonical file sources, then occurrences; ties are lexical. Copies therefore do not increase distinct-text counts. Up to three diverse original context samples are retained. Sampling cannot establish that every occurrence has the same privacy meaning.

For more distinctive suggestions, supply a reference-count JSON object, for example `{"function":100000,"error":80000}`:

```bash
nym terms discover logs.jsonl --background reference-counts.json --limit 300 --output discovery.json
```

Ranking uses Laplace-smoothed background log lift, multiplied by `ln(1 + distinct_texts)`. Foreground weights are distinct-text presence counts, reference weights are supplied counts; vocabularies/casing/units should be comparable. This is a transparent suggestion heuristic, not weighted-log-odds significance, a sensitivity classifier or a calibrated probability. Reference counts are not hard exclusions.

## Scope and safety

Approval means a deliberately authored corpus-wide literal rule, not a propagated occurrence annotation. Choose `contextual` for ambiguous terms such as a name also used as a technical value. Review and discovery fingerprints bind decisions to the exact validated artifact and captured selected-text digest. They do not freeze live inputs: freeze sources yourself for repeatable discovery.

**Exported plaintext lists do not enforce scope.** They apply wherever explicitly passed to `--sensitive-terms-file`. They are not installed globally; review again before using them for another corpus. Normal NYM matching defaults are case-sensitive Unicode word boundaries; substring/case options remain explicit existing controls. Lists cannot veto mandatory detections. Vocabulary decisions must never be treated as complete NER training gold.

Artifacts are staged and published atomically, no-clobber unless `--force`. Files are owner-only on Unix; choose an appropriately private directory/ACL on other platforms. Malformed inputs and resource limits return operational failure, never partial discovery success. SHA fingerprints provide integrity/provenance binding, not authenticated signatures against a malicious artifact author.

## Budgets

Defaults live in `[terms]` in the normal XDG `config.toml`; CLI flags override configured discovery values. Defaults: 50,000 distinct literals, 200 displayed candidates, minimum count 2, three examples, three-word phrases and 16 MiB per line/JSON document or decoded unit. JSON documents are bounded in memory; text/JSONL are streamed. Additional hard limits: 1,000,000 units/tokens/structured spans/source-pairs, 10,000 sources/selected files, 100,000 directory entries, directory depth 64, 5,000 displayed candidates, 128 MiB artifact reads. Configured limits are validated, not silently relaxed.

This first version uses exact bounded counting, not Space-Saving: high-cardinality corpora can fail the distinct/source-pair budget. Increase supported budgets explicitly or use separately reviewed batches; do not interpret limit failures or unshown/singleton candidates as evidence of safety. Throughput and memory depend on vocabulary cardinality and retained contexts. Unknown sensitive material still needs normal detection and evaluation.
