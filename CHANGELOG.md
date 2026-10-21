# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Placeholders can be numbered, with a configurable scope: `--placeholder-scope session` gives `<EMAIL_1>`, `<EMAIL_2>` per class within one input (same value keeps its index), and `--placeholder-scope corpus` gives `<EMAIL_8f3a2b>` derived from the value itself, so sessions processed in any order or in parallel produce identical tokens. Default stays `<EMAIL>`, so existing pipelines keep working; configure with `[replacement] placeholder_scope`. Numbering is what makes placeholders usable as training data: the model learns an identifier shape instead of a plausible-looking fake, and one person is one token across the corpus.

- Corpus deny policy: `corpus inventory --deny-file` evaluates regexes against every source and marks a match as **disqualified** rather than something to redact. `validate`, `apply` and `verify` refuse a plan containing a denied source, so a project that must never be published cannot be "cleaned" into acceptability. Invalid regexes fail closed, the plan records which rule fired, and diagnostics name only the source id and rule index. See `docs/corpus-pipeline.md` for the consolidated pipeline and what it borrows from pi-brain and pi-share-hf.

- Treat infrastructure IPv4 addresses as harmless: loopback, RFC1918 private, link-local, CGNAT (100.64.0.0/10, which covers the tailnet), multicast/reserved and the TEST-NET/benchmark documentation ranges are no longer reported, because they identify infrastructure rather than people. This was the second-largest measured false-positive family: 70% of reviewed `ipv4` findings on real sessions were harmless (96 of 139 distinct values had no planted counterpart: 21 private, 75 public). Public addresses are still redacted; `[detection] report_private_ipv4 = true` restores the old behaviour. Verified: real-session canary coverage unchanged at 990/990 and 375 tests green.

- Ship the vendor/product benign list as an extendable default rather than a hardcoded filter. `DEFAULT_BENIGN_TERMS` (claude opus, gpt-4, mistral large, openai, ...) now applies in person-like classes on **every** profile, so plain `anon`/`detect` stop flagging model names as people (previously only the trace pipeline had this protection). Extend it with `benign_terms` (config, `--benign-terms-file`, profile `allow.txt`); disable it with `--no-builtin-benign-terms` / `builtin_benign_terms = false` when product names are themselves sensitive. Matching is case- and whitespace-insensitive, whole-value only; credentials, identifiers and contact classes can never be suppressed. Trace-policy summary stats are reported only for user-configured policies or recorded outcomes, so default output shape is unchanged.
- Fix GLiNER chunking offsets: chunks were rebuilt with single-space joins and then offset into the original text, so every span in a chunked text (over 200 words, i.e. most real session bodies) pointed at the wrong bytes while still reporting the right matched text. Chunks are now verbatim slices at word boundaries. Found while measuring the GLiNER union; a new unit test fails on the previous join-and-find chunker.
- Cover quoted credential assignments in any key case: `password="..."`, `"password": "..."` and `token: '...'` now match `secret_assignment` (value span excludes the quotes), while bare lowercase assignments and code identifiers such as `password_hash` stay unmatched. SCREAMING_CASE keys additionally accept `:` separators. Closes the repeated-password leak in native text and JSON anonymization (trx-ppar.1).

- Require postal context for NER `postcode` findings: an explicit postal/address cue on the line, or an adjacent capitalized word (city, state, country). Bare numeric constants, ports and prose counts no longer get redacted (a port `8080` in a code line was previously replaced).

- Fix the NER cache on Windows: the append-only cache handle could not be locked (`Access is denied`), so the cache never persisted and every `traces anon` run warned and recomputed. Open the cache file with read access before locking.

- Make the test suite pass natively on Windows 11 (241 unit tests, 19 suites, plus the four cached real-ONNX tests): platform-absolute XDG fixtures, backslash-normalized expected profile paths, verbatim-path-safe artifact writes and temp-home redirection that no longer writes into the real user profile. Document the local `esaxx-rs` static-CRT vendor shim needed to build/link natively on Windows MSVC (see `docs/windows-build.md`); the upstream CRT mismatch remains open (trx-rby8).

- Add `nym corpus inventory/status/schema/validate/apply/verify`: unresolved-by-default structural/opaque-source accounting, explicit session or full scope and identity namespaces, reviewed typed/span/asset/reference transforms, source-bound private ledgers, staged no-overwrite candidates, hash-bound teacher-source handoff and verified resume. Complete declarations are not anonymity or rights certification; native media and semantic/label quality require external review. See `docs/corpus-transformations.md`.

- Fail closed on unreadable, malformed, conflicting, unsupported-version or incompatible replacement maps before anonymization output/model initialization; preserve intentional missing-map first runs. Validate strategy, seed and context before reuse, not only during later persistence.

- Add regression coverage for existing `USERNAME`/`USER_NAME` label aliases and full-value cached-model username findings, including class and byte-offset fidelity.

- Preserve failed macOS test-harness exit statuses when the native teardown hook has no explicit CLI status; retain fast-exit protection for CLI statuses and intentionally leaked native sessions.

- Lower the token NER microbatch budget from 16K to 2K padded tokens to reduce inference working memory. Preserve model, threshold, window context, detection scope and every queued window.

## 0.4.0 - 2026-10-13

- Cap automatic token NER intra-op threads at 8 (macOS performance cores, available CPUs elsewhere) to reduce contention. Tune with positive `ner.threads` or `NYM_NER_THREADS` in layered config; an explicit `--config` file wins. Applies to token detection, CPU fallback and vocabulary ranking, not GLiNER. `config ner-status` reports the effective count.

- `nym traces anon --apply-plan FILE` replaces exactly the reviewed values of a plan everywhere in each session without running detectors (0.2 s instead of a full scan on the 30-session benchmark); kept values protect approved values inside them and uncovered sessions fail. NER results are cached per text in `$XDG_CACHE_HOME/nym/ner` (keyed hashes, offsets and labels only, owner-only, `--no-cache` to disable), so re-scanning a grown session only runs NER on new text (17 s to 1 s on a 514 KB session). Bare "age" findings now need a nearby label, the class of a value found with several classes is deterministic, and escaped `\n` before a value counts as a line break.

- `nym traces anon --progress-json` emits throttled machine-readable progress on stderr (`{"nym_progress":{"phase","done","total"}}` for detect, NER, replace and done), independent of the terminal progress display. pi-nym renders it as a progress bar.

- Ignore a leftover `ner.labels` with `backend = "tokens"` (it has no effect there; older `config init` versions wrote it) with a one-time warning instead of refusing to run. `backend = "both"` still rejects labels, since they would only restrict its GLiNER half.

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

