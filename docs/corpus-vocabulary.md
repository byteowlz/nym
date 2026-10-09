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

The HTML is self-contained and phone-sized, with no uploads or external assets. By default it is an offline file, not a service. For phone access, use the opt-in capability-gated tailnet launcher below or private file transfer; the HTML contains private text. Browser resume is best-effort: download decisions before closing. Reopen with `--resume nym-terms-review.json` to embed saved decisions. Actual iOS browser/download behavior must be checked on the target device.

Machine review also works without HTML:

```bash
nym terms review discovery.json --output review.json --decision CANDIDATE_ID=sensitive
nym terms review discovery.json --resume review.json --output revised.json --decision OTHER_ID=contextual
nym --json terms export discovery.json --review revised.json --output sensitive.txt --dry-run
```

IDs appear in the discovery artifact. Decisions are `sensitive`, `contextual`, `dismiss`, or `unsure`. Only `sensitive` exports; pending, dismissed and uncertain terms never become implicit approvals or benign exemptions. CLI summaries contain counts, not corpus values. JSON/YAML summary output and shell completions use the normal NYM flags.

## Pi-session launcher

From the NYM checkout:

```bash
./scripts/review-pi-terms.sh
# Or choose a project/session subtree and do not open a browser:
./scripts/review-pi-terms.sh --sessions /path/to/sessions --no-open
```

The launcher enables local CPU NER-positive priority at threshold 0.5 before the 200-item display cutoff. It uses the existing cached token model only; no model is downloaded. Recognized literals rank by duplicate-downweighted repetition (so widely shared identifiers are not source-IDF penalized); unmatched candidates retain ordinary source-specific ranking and stay eligible. `--no-ner-rank` disables this boost; `--ner-rank-model DIR` chooses existing local assets. Missing assets or inference errors stop publication rather than silently falling back. Model matches still require human privacy review.

The launcher finds the newest compatible local versioned build (or `nym` on PATH), without installing it. Override with `--binary FILE` or `NYM_BINARY`. It runs native recursive JSONL/NDJSON discovery on `**.text` and `**.thinking`, then creates an offline review page. It uses bounded proposals and exact recount: defaults are 200 suggestions, 200,000 proposal slots and one-word phrases. `--limit`, `--max-distinct`, `--phrase-words` and `--min-count` override these values.

Each run creates an owner-only timestamped directory under `$XDG_STATE_HOME/nym/pi-terms` (fallback `~/.local/state/nym/pi-terms`) with isolated configuration, discovery JSON and review HTML. `--output-dir DIR` chooses a **new** directory; existing directories are never overwritten. It runs inline in ordinary shells, tmux or herdr, never starts a new session, modifies input logs, trains, uploads, globally configures NYM or automatically approves/exports terms. Browser opening is best-effort; `--no-open` disables it. The launcher enables count-only stderr progress for both proposal and exact recount; `--no-progress` disables it. Preserve downloaded decisions separately and use `terms export` after explicit review.

Keep reserved validation sources out of discovery too: choose an appropriate subtree or repeat `--exclude-file FILE`. The launcher does not know your holdout protocol and does not bypass counting/traversal limits. It is a Bash launcher, tested with macOS system Bash; Windows users can use the native CLI directly.

## Saved runs in the native CLI

```bash
nym terms runs list
nym --json terms runs list
nym terms runs show latest
nym terms runs open latest
nym terms runs serve latest --tailnet --open
# Or use a unique ID prefix shown by list:
nym terms runs serve RUN_ID --tailnet --port 8521
nym terms runs register /private/completed-run
```

Mutable inventory manifests live at `$XDG_STATE_HOME/nym/terms/runs` (fallback `~/.local/state/nym/terms/runs`). Each completed run gets its own private, atomically published manifest, containing a canonical-directory-derived ID, creation time, counts and artifact fingerprints—not contexts or server tokens. Registration is idempotent; changed content requires explicit `--force`. Custom output directories can be registered too. The Pi launcher registers only after both discovery and HTML publication succeed; registration failure stops automatic opening/sharing.

Existing `$XDG_STATE_HOME/nym/pi-terms` directories appear automatically as legacy entries without moving files or reading contexts. Listing is metadata-only; `ready` means both regular artifact files exist, **not** integrity or privacy certification. Incomplete/missing runs remain visible. `latest` chooses the newest available pair, while explicit IDs accept unambiguous hexadecimal prefixes of at least four characters. Corrupt manifests, ambiguous IDs and inventory budgets (10,000 entries, 64 KiB per manifest) fail without partial machine output.

Opening/serving validates the entire bounded discovery and source-bound review, and rejects changed registered artifacts rather than silently choosing an older run. Desktop opening creates a private trusted-UI snapshot in `nym/terms/views`, discarding supplied scripts. Serving installs hash-verified, compiled-in public helper/UI assets under `$XDG_DATA_HOME/nym/terms/review-host` (OS user-data fallback when unset, `~/.local/share` on Linux); no checkout is required. A checksum is rechecked before the host snapshots HTML, so a verification/read race cannot publish changed content. `[paths].state_dir`/`data_dir` override defaults and expand `~` and environment variables. Runtime/OS data-directory fallbacks apply when XDG variables are absent.

Native serving defaults to loopback; `--tailnet` is explicit. It still requires `uv` only when serving, does not rescan source sessions, enables no model, preserves browser-only downloaded decisions and never installs itself or changes Tailscale/Funnel. `--json`, `--yaml` and `--quiet` work for inventory; serving JSON/YAML includes a **private capability URL**, not public metadata. Keep its terminal running; Ctrl-C stops the foreground host.

## Private phone sharing

```bash
# Discover, then share the new review; print a private phone link:
./scripts/review-pi-terms.sh --serve
# Reuse an existing run without rescanning or modifying its files:
./scripts/review-pi-terms.sh --serve-existing "$RUN" --no-open
```

Serving alone requires `uv` and a running Tailscale client; offline discovery does not. `--port N` overrides 8521. The process stays in the current terminal/tmux/herdr pane; **Ctrl-C stops sharing**, preserving review artifacts. `--no-open` suppresses opening the desktop browser. No staging directory, public Funnel, global Tailscale configuration or model is used. Startup failure never falls back to a public listener.

Only the selected regular `review.html` is read, once. The helper extracts its NYM data and rerenders the trusted repository UI, discarding supplied HTML/scripts; it never serves the directory, discovery JSON, configuration or unrelated files. It bounds page reads to 128 MiB, active connections to eight and request headers to 8 KiB/two seconds. This is a read-only snapshot, not a server-side annotation store or accuracy validation.

The printed URL is a **private capability**: anyone with both tailnet reachability and the link can view its embedded contexts. Its random token is in the fragment, removed before authenticated fetching—not in request paths, queries or access logs. Responses are no-store, no-referrer, frame-denied and restricted by CSP; no request logs are emitted. HTTP is carried over the encrypted tailnet, not public HTTPS. Do not publish the link. Decisions stay in the browser tab; download review JSON for durability and export explicitly. Restarting creates a new capability. Real iOS Safari ergonomics remain to be checked.

For non-Pi reviews, the same optional helper supports `uv run --no-project python scripts/serve_terms_review.py review.html --tailnet`; it defaults to loopback without `--tailnet`. Native `nym terms` remains server-free.

## High-cardinality corpora: bounded proposals

Exact discovery remains the default and fails closed when the distinct-literal budget is exhausted. For large histories, use `--bounded` (enabled by the Pi launcher):

```bash
nym terms discover /path/to/sessions -r --extension jsonl,ndjson --bounded \
  --include '**.text' --include '**.thinking' --phrase-words 1 \
  --max-distinct 200000 --limit 200 --output discovery.json
```

Pass one uses deterministic Space-Saving occurrence-frequency proposals with at most `--max-distinct` slots. It keeps no example contexts, source-pair sets or global duplicate-text set. Lower-bound estimated frequencies select an oversampled pool of at most 10 times `--limit`. Pass two scans the same resolved file set and computes **exact occurrences, distinct-text counts, source counts, contexts and final spread ranking for that pool only**. Source counts use compact bounded bitsets. Repeated text copies are downweighted exactly in final ranking, but can influence first-pass proposals. Estimated frequencies are discarded; they are never presented as exact review counts.

The artifact records `selection.method`, proposal capacity and pool size; `candidate_count` then means **recounted pool**, not total corpus vocabulary. The CLI and offline review explicitly identify this mode. Omitted terms—including rare sensitive values—remain unmeasured: approximate shortlist selection is not exhaustive privacy detection or training gold. It is not guaranteed equivalent to exact discovery. Bounded mode currently requires named files/directories (not stdin) and does not support background ranking.

The selected-unit/source/path/order checksum must match across passes or no artifact is published. The resolved file set is fixed before scanning; file bytes are not snapshotted, and changes outside selected units are not covered by this checksum. Freeze live session files separately for a reproducible corpus. Malformed/unreadable input and other resource failures still fail closed. Bounded mode supports at most 10 million units; exact mode retains its one-million-unit limit. Exact duplicate accounting still consumes memory proportional to distinct text hashes, bounded by the unit limit; only candidate proposal storage is independent of total vocabulary cardinality. This is not an arbitrary-size or constant-total-memory claim.

## Optional local NER priority

Native discovery stays model-free by default. Add `--ner-rank` to recognize **bare literals in the entire eligible exact/recounted pool**, before `--limit` truncation. It uses the configured token model or the cached default small INT8 model, with CPU, threshold 0.5 and ordinary argmax decoding; regex, decision services and literal policies do not participate. `--ner-rank-model DIR_OR_CACHED_REF` overrides the model without downloads. A build with the `ner` feature is required.

Positive matches come first, ordered by exact distinct-text counts, with original ranking as the tie-break; unmatched candidates retain their previous ranking. This is suggestion priority only: no match is not benign, no match auto-approves anything, and omitted proposal terms are still unmeasured. The artifact binds `ner_ranking` method, model/config/tokenizer file SHA-256s, threshold and scored/recognized counts; each displayed candidate has `ner_match`. Asset checks repeat after loading/scoring. These file hashes do not attest auxiliary files of third-party external-data ONNX models. Source-bound reviews/export remain explicit; old model-free artifacts retain their format/fingerprints.

## Source-specific suggestions

`terms discover --specific` ranks repeated literals by `ln(1 + distinct_texts) × ln((source_count + 1) / (term_sources + 1))`. Terms spread across almost every source fall below repeatedly mentioned, more localized vocabulary. Counts are still exact for the recounted pool; ranking is not a privacy classifier. With one contributing source there is no source contrast, so ranking falls back to logarithmic distinct-text frequency. Background ranking is a separate, mutually exclusive option.

Bounded specificity widens the proposal/recount pool from 10× to 100× the display limit, capped by `--max-distinct`, and records `selection.pool_factor: 100`. This costs more recount memory/time and gives localized terms more opportunity to enter review, but occurrence-heavy proposals can still omit rare terms. It is not exhaustive. No dictionary, embeddings, external service or automatic benign exemption is added.

The Pi launcher now explicitly requests this mode. `--frequency-ranking` restores its old spread ranking and 10× proposal pool. Existing native discovery defaults and older artifacts/reviews remain unchanged. Widely shared personal names, usernames or other sensitive values can rank low; omitted literals remain unmeasured, not safe. Human review must still decide whether any displayed literal is private. Review source-specific output before considering embeddings or a language-model triage layer.

## Explicit Unicode projection

`terms discover --repair-unicode` accepts unpaired UTF-16 `\uXXXX` escapes in **JSON string values** by replacing each invalid code unit with U+FFFD in a discovery-only decoded view. Strict native discovery remains the default. The Pi launcher enables this explicitly; use `--strict-unicode` there to reject instead. Input session files are never rewritten, skipped or uploaded. This option does not affect detection/redaction/restoration or gold preparation/training.

Valid surrogate pairs, ordinary Unicode (including existing U+FFFD), literal/double-escaped `\\ud800` text and other JSON escape semantics remain unchanged. Non-Unicode grammar errors, invalid UTF-8 bytes and parser/resource limits still fail closed. **Unpaired surrogates in object keys are rejected**, not repaired: changing keys could alter selectors or collapse distinct keys.

Artifacts and summaries include `projection.policy`, repaired-record/replacement counts and selected units from those records. Every retained example from a repaired record is conservatively marked `projected`, even when its own string did not change. The offline review labels these contexts and warns that offsets, unit hashes, counts and duplicate-text grouping refer to the **decoded scalar view**, not original UTF-16 values. Different invalid original values can project to identical strings and be grouped as identical view text. Do not use this view as original-offset NER training gold.

The corpus fingerprint also incorporates each repaired record's **original parser-input bytes** (including framing/BOM), source identity, physical location and projection policy. Bounded proposal/recount rejects changed original repaired bytes even if their projected strings are equal; prior reviews cannot silently apply to a different original corpus. For ordinary records the existing selected-unit checksum contract remains unchanged. Exports still require explicit human approval and do not automatically enforce corpus scope.

## Progress

Interactive `terms discover` shows a count-only bar on stderr; the launcher explicitly enables it. `--progress` also requests stage-start/end count lines when stderr is redirected. `--no-progress` and global `--quiet` suppress it. JSON/YAML stdout stays machine-readable. Minimal-feature builds support the same behavior without an extra dependency.

Each bounded pass has its own file total, selected-text-unit count and elapsed time. Units update even inside a large file (terminal redraws are throttled to eight per second). The percentage measures **files scanned**, not bytes, estimated remaining time, privacy coverage or successful publication. Stdin is treated as one input stream. Failed reads/parses mark the current pass incomplete; finishing a scan does not claim the destination was published. Bars contain no session paths, IDs or text.

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

Traversal has a `--max-files` budget (default/hard ceiling 10,000 unique selected files), 100,000 inspected directory entries and 64 nested directories. Failed traversal or extraction never publishes a partial deck or overwrites a destination. Successful JSON/YAML and human summaries report selected files and skipped extension/symlink/artifact counts without paths or corpus values; bounded results also report proposal/recount scope. Discovery vocabulary/unit/source-pair limits below remain separate: recursive input does **not** promise arbitrary whole-history scalability.

## Inputs and statistics

Supports text, JSON and JSONL/NDJSON, multiple named files, recursive directories or stdin. Structured JSON traversal honors the parser's still-enabled 128-level recursion budget; it no longer rejects parser-accepted data at 64 levels. This is separate from the 64-level filesystem directory budget. Binary documents must first be extracted to supported text; this command does not yet reuse PDF/office/OCR extraction. `--format` overrides conservative format resolution. JSON string values are decoded, not raw escape sequences. `--include`/`--exclude` use the existing NYM selectors, applied within each JSONL record. Text units are physical lines; offsets are UTF-8 bytes within the original decoded unit, not the complete input file. JSONL examples retain physical `record[N]` locations, including blank-line numbering.

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
