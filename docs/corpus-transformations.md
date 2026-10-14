# Reviewed corpus transformations

`nym corpus` inventories and transforms a working source tree before teacher labeling. It does not decide rights, inspect image pixels, certify anonymity, fix segmentation, train models, or publish anything. Keep diagnostic corpora/checkpoints private and unchanged. Use a separate working input and a small externally reviewed pilot before full regeneration.

## Workflow

```bash
nym corpus inventory WORKING_INPUT -o PRIVATE_PLAN.json --scope session
nym corpus status PRIVATE_PLAN.json --json
nym corpus schema > corpus-schema.json
# Review/edit PRIVATE_PLAN.json against immutable working sources.
nym corpus validate PRIVATE_PLAN.json --source WORKING_INPUT
nym corpus apply PRIVATE_PLAN.json --source WORKING_INPUT \
  -o CANDIDATE --ledger PRIVATE_LEDGER.json --dry-run
nym corpus apply PRIVATE_PLAN.json --source WORKING_INPUT \
  -o CANDIDATE --ledger PRIVATE_LEDGER.json
nym corpus verify PRIVATE_LEDGER.json --source WORKING_INPUT --bundle CANDIDATE
```

Plans and ledgers contain originals and private hashes/context. Keep them outside input and candidate trees, outside git, and never distribute them. Files are owner-only. Candidate output is also private by default; publishing remains a separate authorized action. Existing destinations are never overwritten. `apply --resume` verifies an existing matching bundle or reconstructs an interrupted ledger-only commit. Different plans, source changes and corrupt artifacts block resume.

## Scope and decisions

Session and full scopes are peer choices, not quality levels:

- `session` (default): each file starts in its own `group`. Set equal groups explicitly when a session spans messages, memory and tool files.
- `full`: aliases are shared across the whole reviewed corpus.
- Optional unit `entity` distinguishes identical literals with different meanings or identities. Use the same namespace for linked occurrences; namespaces do not establish identity automatically.

Every unit starts `unresolved`. Neither discovery omissions, no findings, an allowlist, nor ownership establish safety. Every source needs an explicit `review` with declared `basis` (`owned`, `licensed`, `synthetic`), external `evidence`, `privacy_review`, and `serialization_reviewed`/`media_reviewed` acknowledgments. These are operator declarations, not nym verification. Every unit needs a nonempty `approval` receipt and an action:

```json
{"kind":"retain"}
{"kind":"literal","value":"reviewed synthetic value"}
{"kind":"fake","pattern":"first_name"}
{"kind":"spans","edits":[{"start":0,"end":5,"action":{"kind":"fake","pattern":"first_name"}}]}
{"kind":"asset","path":"/private/owned-replacement.png","sha256":"<replacement-byte-sha256>"}
{"kind":"reference","source_id":"f00001"}
```

Offsets address UTF-8 bytes in the decoded unit, not source-file bytes or character indexes. Span edits must be ordered, nonoverlapping and on character boundaries. A unit approval also explicitly reviews retained remainder. Retained spans can anchor known unchanged facts. References target an accounted source's final filename, relative to the bundle's `sources/` root; consumer adapters must honor that root. They are not arbitrary HTTP URLs. Filenames permit retention or literal relative renames only.

Fake generation reuses nym's local replacer without inference or downloads. Scoped repeated values must agree; different transformed values/identity namespaces must not collapse onto one alias or a retained value. JSON key and filename collisions block output. Supported structured fakes have format checks; IBAN surrogates preserve country, length, separators and mod-97, and card surrogates require Luhn. Name fakes that change component count or remove a Unicode/script challenge are rejected; use coordinated reviewed literals where automatic fidelity is unsupported. This does not verify arbitrary semantic relationships, geographic consistency or extraction difficulty.

Opaque/native files require explicit externally reviewed retention or a hash-bound replacement asset; never automatic stripping. Embedded image/media `data` strings in recognized JSON objects require external media review and whole, valid-base64 retention/replacement. No pixels, audio, hidden document metadata or remote referenced resources are inspected. URLs and unknown embeddings remain reviewed string content or opaque source work; use externally prepared owned/licensed sources and adapters when needed. A text-only review is not native-media approval.

## Accounting and handoff

Inventory includes every file, its relative filename, JSON keys and string/numeric/boolean/null values, empty containers, structural strings, blank JSONL records and opaque assets. Duplicate keys (including escaped equivalents), malformed/unpaired Unicode, symlinks, empty directories and special filesystem entries are rejected rather than silently omitted. Unsupported extensions are whole opaque assets, not selectively inspected documents. Bounds: 10,000 files, directory depth 32, JSON depth 64, 16 MiB/input file and 64 MiB/input tree; oversized work blocks rather than truncates. Outputs are bounded to 64 MiB/file and 256 MiB/tree.

JSON is re-encoded; numeric lexemes remain exact (including integers beyond floating-point precision). Whitespace and escape spelling can change, and JSONL newlines are normalized; approval must acknowledge this. Plain text is not JSON-quoted. These changes may affect extraction difficulty and must be evaluated in the pilot.

Private v1 ledgers retain source snapshots, decisions and actual unit outputs. Verification reconciles every source/unit, regenerated output, manifest and bundle file; it rejects extra, missing, modified or stale records. Late validation failures publish no candidate bundle. The public `manifest.json` contains only approved output paths, aggregate declarations and actual transformed-source hashes/lengths, not original hashes or mappings. Teacher labels must bind those transformed source hashes, never hashes of private originals.

A candidate with complete declared decisions is not a release clearance. Independently review source rights/confidential facts/code/media, segmentation/accounting, complete sensitive-value removal, native evidence, contradictions/supersession and downstream label quality. Train release candidates from a clean base only after those gates; sanitized fine-tuning cannot erase an already contaminated checkpoint.
