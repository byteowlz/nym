# Reviewed corpus transformations

`nym corpus` inventories and transforms a working source tree before teacher labeling. It does not decide rights, inspect image pixels, certify anonymity, fix segmentation, train models, or publish anything. Keep diagnostic corpora/checkpoints private and unchanged. Use a separate working input and a small externally reviewed pilot before full regeneration.

## Workflow

```bash
nym corpus inventory WORKING_INPUT -o PRIVATE_PLAN.json --scope session \
  --deny-file deny.txt   # optional: a match disqualifies the source
nym corpus status PRIVATE_PLAN.json --json
nym corpus schema > corpus-schema.json
# Review/edit PRIVATE_PLAN.json against immutable working sources.
nym corpus validate PRIVATE_PLAN.json --source WORKING_INPUT
nym corpus apply PRIVATE_PLAN.json --source WORKING_INPUT \
  -o CANDIDATE --ledger PRIVATE_LEDGER.json --dry-run
nym corpus apply PRIVATE_PLAN.json --source WORKING_INPUT \
  -o CANDIDATE --ledger PRIVATE_LEDGER.json
nym corpus verify --ledger PRIVATE_LEDGER.json --source WORKING_INPUT --bundle CANDIDATE
```

Plans and ledgers contain originals and private hashes/context. Keep them outside input and candidate trees, outside git, and never distribute them. Files are owner-only. Candidate output is also private by default; publishing remains a separate authorized action. Existing destinations are never overwritten. `apply --resume` verifies an existing matching bundle or reconstructs an interrupted ledger-only commit. Different plans, source changes and corrupt artifacts block resume.

## Proposing decisions from reviewed values

Deciding every unit by hand works for a handful of files. For agent sessions, let
detection and value review do the work and approve the result in bulk:

```bash
nym traces anon WORKING_INPUT --plan PRIVATE_VALUES.json   # detection, NER, profiles
# review the values (pi-nym, scripts/review_trace_anon.py); remove false positives
nym corpus inventory WORKING_INPUT -o PRIVATE_PLAN.json --scope session
nym corpus propose PRIVATE_PLAN.json --source WORKING_INPUT \
  --values PRIVATE_VALUES.json -o PRIVATE_PROPOSED.json [--rename-files]
nym corpus approve PRIVATE_PROPOSED.json -o PRIVATE_APPROVED.json \
  --receipt "values reviewed by <who> on <date>" --all-classes --unflagged --renames
nym corpus review PRIVATE_APPROVED.json -o PRIVATE_REVIEWED.json --basis owned \
  --evidence "<rights statement>" --privacy-review "<who, how, when>" --serialization-reviewed
```

- `propose` reads the reviewed values per session (the `traces anon --plan`
  format, keyed by the path relative to the input directory) and matches them
  exactly like `traces anon --apply-plan`. Every occurrence in a string or key
  becomes a `placeholder` edit, `<EMAIL_1>`, numbered per identity namespace: per
  file with `session` scope, across the corpus with `full` scope. One value gets
  one class per namespace (the first in plan order), so it never gets two tokens.
- By default a reviewed value is replaced in every covered source, not only in the
  session it was detected in (`--values-scope corpus`): a name caught in one
  session is sensitive in the next one too, where detection may have missed it.
  In the 20-session pilot, values listed for one session occurred 8,753 times in
  others. `--values-scope session` restricts each value to its own session. This
  is independent of placeholder numbering, which follows the plan `scope`.
- `--replace-field KEY` (repeatable or comma-separated) replaces every string under
  that JSON key with one placeholder, for structure that detection skips on purpose
  but publication must not keep. For agent sessions, use
  `thinkingSignature,textSignature,id,parentId,toolCallId,responseId`: signatures
  are the provider's encrypted copy of the reasoning (the pilot's reviewer flagged
  them, and they were 17.5% of all retained characters), and ids link to provider
  logs. Equal values share one token and one class, also in units under other
  keys, so `toolCallId` still matches its call and `parentId` its parent.
- `--strip-media` replaces embedded PNG and JPEG images with a neutral 1x1 image of
  the same type. Replaced images need no media review; retained ones still do.
- Each changed unit records a `proposal` with its basis: `values` (placeholders),
  `unflagged` (no reviewed value occurs, retain), `rename` (opaque file name
  `f00003.jsonl`; always used when a name contains a value, for every file with
  `--rename-files`), `field` (whole value under or equal to a replaced field),
  `media` (embedded images, retention proposed but never inspected),
  `media_replaced` (neutral image), `manual` (a value occurs in a number or other typed unit, which
  only takes a literal) and `uncovered` (the values file has no entry for the
  source; nothing is proposed). Units that already have a decision are left alone.
- A proposal is not an approval. `approve` writes the receipt on proposals you
  select by class (`--class email`, repeatable, or `--all-classes`) and by basis
  (`--unflagged`, `--renames`, `--fields`, `--media`). Unselected, `manual` and `uncovered`
  units stay pending, and the summary reports what remains by basis and class,
  plus any `--class` that matched nothing (usually a typo).
- `review` writes one source review on every source that has none.
- `unflagged` means no *reviewed value* occurs there, not that the text is clean:
  values detection missed are retained. Sample unflagged units before approving
  them, and run detection on the candidate bundle as a residual check.
- `propose`, `approve` and `review` write a new plan each time and never
  overwrite, so the sequence of plans is the audit trail. `plan.proposed` records
  the digest of the values file the proposals came from.

## Scope and decisions

Session and full scopes are peer choices, not quality levels:

- `session` (default): each file starts in its own `group`. Set equal groups explicitly when a session spans messages, memory and tool files.
- `full`: aliases are shared across the whole reviewed corpus.
- Optional unit `entity` distinguishes identical literals with different meanings or identities. Use the same namespace for linked occurrences; namespaces do not establish identity automatically.

Every unit starts `unresolved`. Neither discovery omissions, no findings, an allowlist, nor ownership establish safety. Every source needs an explicit `review` with declared `basis` (`owned`, `licensed`, `synthetic`), external `evidence`, `privacy_review`, and `serialization_reviewed`/`media_reviewed` acknowledgments. These are operator declarations, not nym verification. A source whose transformed bytes equal its input bytes also needs `publish_verbatim: true`: retaining every unit publishes the original file, so that is a decision rather than a side effect. Every unit needs a nonempty `approval` receipt and an action:

```json
{"kind":"retain"}
{"kind":"literal","value":"reviewed synthetic value"}
{"kind":"fake","pattern":"first_name"}
{"kind":"placeholder","class":"email"}
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
