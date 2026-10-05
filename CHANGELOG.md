# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Create reversible key files through private atomic temp files; reload and validate one-to-one mappings under a single crash-safe write lock.
- Reserve imported/generated fake aliases and preserve case-distinct originals in seeded resumed runs; use a bounded unique-placeholder fallback for exhausted fake domains.
- Validate keys before deanonymization and restore in one pass, prioritizing full aliases and skipping ambiguous component mappings instead of mutating restored originals.
- Support recursive `**` JSON selectors, including zero-depth matches and subtree exclusions.
- Reject unknown audit policies and correctly gate configured/NER findings; keep JSON coverage on stderr and report streaming JSONL coverage.
- Initialize ORT before CoreML hardware probes; support token-NER `[ner] provider = "cpu"` and CPU fallback after accelerator session errors.
- Validate synthetic benchmark annotations, score full/partial UTF-8 spans and actual benign preservation separately, and record fresh revision/content hashes.
- Restrict low-confidence username detection to conventional or explicitly labeled accounts to avoid matching code tokens.

