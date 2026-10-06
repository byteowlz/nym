# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

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

