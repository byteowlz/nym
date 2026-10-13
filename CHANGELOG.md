# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- `nym traces anon --plan FILE` reports every value that would be replaced (class, detected/listed/propagated source, credential flag, occurrence count, short context) without writing outputs; `--keep-values-file` applies explicit keep decisions, also to credentials and identity terms. Enables interactive review such as pi-nym.

- Resolve XDG base directories per the Base Directory Specification : unset, empty or relative `XDG_CONFIG_HOME`/`XDG_DATA_HOME`/`XDG_STATE_HOME` values fall back to `~/.config`, `~/.local/share` and `~/.local/state` on Linux and macOS (previously macOS used `~/Library/Application Support` and invalid values were used as-is), or `%APPDATA%`/`%LOCALAPPDATA%` on Windows, matching the rust-cli template. The model catalog cache, build store and Pi launcher follow the same rule.

- Add redaction profiles (`nym profiles list/show/add/remove/import/path`): private `sensitive.txt`/`allow.txt` lists per named profile under the config directory, `default` always applied, others stacked with `--redaction-profile` in `anon`, `detect` and `traces anon`. Replaces the single `trace-allow.txt` file.

- `nym traces anon`: keep images by default (`--strip-images` opts in); trim delimiters NER attaches to entity edges so one value keeps one pseudonym (`USER=wismut ;`); never replace system accounts; add trace precision rules and a built-in plus user allowlist (`--allow-terms-file`, `trace-allow.txt`) that cannot veto credentials, identity or sensitive terms. On the 30-session benchmark distinct replaced values fell from 1,673 to 1,036 with 660/660 inserted values still removed.

- Add `nym traces anon` for agent session JSONL: parallel per-session processing with session-consistent, cross-session-independent pseudonyms, optional per-session key files, a private side-by-side review page (`scripts/review_trace_anon.py`), local-identity literals, session-wide propagation of every detected value, structure-preserving home paths, unscanned opaque/image fields with opt-in `--strip-images`, NER limited to prose and name-signal lines by default, and numeric shape checks against timestamp/ID false positives. On 30 real Pi sessions with inserted test values: 660/660 removed and 0 of 3,367 owner-identity occurrences left; 15x faster than per-file `nym anon --ner` on a shared subset.
- Detect vendor-prefixed tokens (Anthropic, OpenAI, GitHub, GitLab, Slack, Google, Hugging Face, Stripe, ...), environment-style secret assignments and PEM private keys at high confidence.
- Window batched token NER like single-text detection, fixing unbounded padded tensors (100+ GB RSS) on long inputs and applying secret-token expansion in batches too. Map the model's `USERNAME` label to `username`.
- Preserve JSON object key order in anonymized JSON/JSONL output.

- Add optional local CPU `terms discover --ner-rank` before the review cutoff, using only cached/local token models at threshold 0.5. Prioritize recognized values by exact duplicate-downweighted repetition; keep unmatched values eligible, preserve ordinary ties, bind scorer file hashes/counts, and fail closed without downloads or privacy auto-decisions. Enable it in the Pi launcher with `--no-ner-rank` and `--ner-rank-model` controls; native model-free discovery remains the default.

- Expose saved vocabulary runs with native `terms runs list/show/register/open/serve`, stable IDs/`latest`, private per-run XDG-state manifests and metadata-only legacy discovery. Register completed launcher runs, verify source/artifact scope before reuse, render trusted desktop snapshots, and embed hash-checked XDG-data hosting assets for checkout-free private serving; no rescan, model or automatic installation.

- Integrate opt-in private tailnet review sharing into the Pi launcher (`--serve`, `--serve-existing`, `--port`): read-only bounded snapshots, fragment capabilities, no directory exposure/request logs/public Funnel, foreground Ctrl-C shutdown and browser-only decisions. Offline/native workflows remain unchanged; serving alone uses uv/Python stdlib.

- Add opt-in `terms discover --specific` source-IDF ranking and a wider bounded recount pool; use it in the Pi launcher to reduce ubiquitous-word dominance, with `--frequency-ranking` restoring old behavior. Suggestions still require human privacy decisions and omissions remain unmeasured.
- Add explicit discovery-only `--repair-unicode` for lone UTF-16 escapes in JSON values; preserve original logs, bind repaired original-record bytes into corpus provenance, label projected contexts/view-relative offsets and keep object-key/grammar errors fail-closed. Enable it in the Pi launcher with a `--strict-unicode` opt-out; native detection/redaction/gold stay strict.
- Align vocabulary JSON traversal with the parser's guarded 128-level budget instead of rejecting valid structured session data beyond 64 levels; retain fail-closed parsing and the separate directory-depth limit.
- Show count-only stderr progress for vocabulary discovery and both bounded passes, including file totals, text-unit counts and elapsed time; support `--progress`, `--no-progress` and quiet mode. Enable it in the Pi launcher without changing machine-readable stdout.
- Add bounded vocabulary proposals with exact corpus recount (`terms discover --bounded`) to avoid distinct-vocabulary overflow on large histories; use it in the Pi launcher and label approximate shortlist scope explicitly. Preserve exact counts, duplicate downweighting, replay checks and failure-safe publication.
- Add `scripts/review-pi-terms.sh` to discover Pi-session vocabulary with a compatible local build and private durable offline-review artifacts, without installation, session changes or training.
- Discover corpus terms directly from recursive directory trees (`--recursive`, `--extension`, `--exclude-file`, `--max-files`) with deterministic deduplicated input sets, no symlink following, bounded traversal and failure-safe publication; report skipped-file counts without source values.
- Add model-independent `terms discover|review|export` for bounded corpus vocabulary ranking, original-context review, offline phone-sized decisions and explicitly approved private literal lists; dismissal never becomes a benign exemption. See `docs/corpus-vocabulary.md`.
- Prepare source-disjoint complete NER gold bundles, loss-mask all uncertain overlaps and reject unknown labels/truncated gold; add an isolated model-only full-value selection gate with actual-output reconciliation. Tooling does not train or promote a model automatically. See `docs/ner-gold.md`.
- Clear inherited fixed tokenizer padding before token NER windowing, so published BERT/DistilBERT exports do not turn short inputs into invalid padding-only source windows. Batched inference retains its explicit padding.
- Expand token-model password and API-key spans to the whole secret token (stopping at whitespace, quotes, brackets and key/value delimiters), so context-dependent sub-word decoding no longer leaves part of a secret in the output.
- Detect compact and space-grouped IBANs by country length and mod-97 checksum at high confidence, so default scans redact the whole IBAN instead of a phone/card digit fragment. Accept build-stamped `--version` output in benchmark provenance checks.
- Add an opt-in agent-trace privacy profile, guarded literal lists and value-free policy counts; preserve default detection and protected sensitive findings.
- Verify residual pseudonyms by default before staged text/JSON/JSONL/office restoration; restore full exact aliases only, with an explicit `--no-verify-restore` legacy opt-out.
- Add exact-labelled disjoint trace corpora, local-only canary/stratified review tooling, trainable small-checkpoint recovery and advisory local System-1 safety tests; tooling success is not a production-quality claim. See `docs/agent-trace-policy.md`.
- Embed source revision, dirty state, features and target in build versions, with isolated versioned local build/install provenance.
- Preserve original UTF-8 byte offsets in batched token NER, including leading Unicode whitespace; verify cached CPU batch/single parity without downloading models.
- Automatically parse JSONL/NDJSON and bounded, replay-safe stdin/extensionless structured input; preserve string selectors, scalar types, ordering, audit policies, and reversible aliases. Restore JSON/JSONL aliases after decoding strings so quotes, newlines, and backslashes remain valid. Stage structured results and streaming destinations to avoid clobbering files on failure; see `docs/structured-input.md`.
- Fail closed on requested NER initialization, record/chunk inference, and batch failures (including native-wrapper panics), including both backends; preserve operational and policy exit codes through the macOS ONNX exit hook.
- Resolve NER model overrides by backend, reject ambiguous/incompatible controls, validate explicitly GLiNER-only labels, and expose redacted effective settings with `config ner-status`.
- Add an offline synthetic agent-trace precision/utility gate with independently validated occurrence annotations, retained-output checks, held-out cases, and optional cached-model comparisons. Redact contextual account values without consuming their field labels or quotes; expose UTF-8 byte offsets in structured findings.
- Create reversible key files through private atomic temp files; reload and validate one-to-one mappings under a single crash-safe write lock.
- Reserve imported/generated fake aliases and preserve case-distinct originals in seeded resumed runs; use a bounded unique-placeholder fallback for exhausted fake domains.
- Validate keys before deanonymization and restore in one pass, prioritizing full aliases and skipping ambiguous component mappings instead of mutating restored originals.
- Support recursive `**` JSON selectors, including zero-depth matches and subtree exclusions.
- Reject unknown audit policies and correctly gate configured/NER findings; keep JSON coverage on stderr and report streaming JSONL coverage.
- Initialize ORT before CoreML hardware probes; support token-NER `[ner] provider = "cpu"` and CPU fallback after accelerator session errors.
- Validate synthetic benchmark annotations, score full/partial UTF-8 spans and actual benign preservation separately, and record fresh revision/content hashes.
- Restrict low-confidence username detection to conventional or explicitly labeled accounts to avoid matching code tokens.

