# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

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

