# Structured input and scan failures

`nym anon trace.jsonl` and `nym detect trace.ndjson` parse records automatically.
Extensions `.json`, `.jsonl`, and `.ndjson` are case-insensitive. `--format text`,
`--format json`, and `--format jsonl` override detection (`ndjson` is an alias).
JSON means one document, including pretty-printed arrays/objects; JSONL means one
JSON value per nonblank line. Declared malformed structured input is an error,
never a fallback to plaintext.

Stdin and extensionless input use a replayed prefix of at most 64 KiB. Valid
object/array syntax selects JSON; a complete first-line value followed by more
content selects JSONL. Plaintext/code without recognizable JSON syntax stays
text. A singleton record without an extension is a JSON document. Ambiguous or
very long prefixes need `--format`; sniffing cannot prove a format universally.
An initial UTF-8 BOM, leading whitespace, CRLF, and blank JSONL lines are accepted.
Other named extensions stay text. Binary office/PDF/image inputs retain their
dedicated handlers; textual `--format` overrides on them fail explicitly.

## Record semantics

Only decoded string values are scanned. Keys and numeric/bool/null types remain
intact. Include/exclude paths apply relative to **each record**, including tool
arguments and thinking fields; metadata is not automatically exempt. Findings
and coverage identify physical lines as `record[N].path`, with string-relative
match offsets. JSONL anonymization emits compact, valid JSON per nonblank record
in original order. Consistent/fake strategies and key files reuse aliases across
records. `deanon` uses the same format resolver and restores decoded string values,
then safely re-escapes originals (including quotes, newlines, and backslashes).
Keys and non-string scalars are never rewritten by restoration. Structured
processing may normalize outer JSON formatting; reversal preserves data, not the
original serialized whitespace. An empty JSONL file is a complete scan with zero
records.

Restoration now defaults to strict residual verification: only full exact aliases
are restored, while recognizable components/case variants and aliases in unchanged
keys block staged publication. `--no-verify-restore` explicitly selects legacy
restoration. See [trace policy and restoration limits](agent-trace-policy.md).

Automatic JSONL does not silently enable `--stream`. Normal detection supports
`--fail-on`, summaries, selectors, and coverage. Summary JSON contains counts and
blocker classes, never values or paths. Exit codes: **0** complete inspection,
**2** complete audit with policy blockers, **1** operational/incomplete scan or
invalid policy. Operational errors never emit a clean aggregate manifest.

## Output and memory guarantees

Normal JSONL scans process one record at a time, with a 16 MiB physical-line limit.
Sniffing is bounded to 64 KiB; record parsing uses memory proportional to the
largest permitted record. Aggregate detection retains counts, not all findings;
full findings and anonymized payloads spool to owner-only temporary files.
Consistent/fake alias caches grow with distinct sensitive values. Requested key
files retain replacement records and therefore need memory proportional to those
records; this is not a constant-memory key export.

Normal JSONL stdout is withheld until the whole scan succeeds. Output files are
staged beside the destination and atomically replaced only after successful
processing and requested key persistence. Malformed later records or failed NER
leave an existing destination unchanged. A key file and output file are separate
atomic publications, not a transactional pair; a final output I/O failure can
occur after the key was extended. Stdout publication itself can fail partway.

Explicit `--stream` detection and text anonymization may emit already completed
records/units before a later error. Treat **any exit 1 as incomplete and discard
that stdout**. They never emit the failed unit. Streaming output files remain
staged. JSONL anonymization uses the same complete-scan staging even with
`--stream`; single JSON documents remain buffered. Streaming audit policies,
summaries, detection coverage, and detection `--json`/`--yaml` output are explicitly
rejected; omit `--stream` for
a complete bounded JSONL audit. Use `--format jsonl`, not `--format json`, for
multiple records, including with `--stream`. Streaming and normal scans share
ruleset, pattern, confidence, and NER configuration resolution.

## NER coverage

Requested/configured NER must initialize and finish every selected backend,
record, chunk, and batch. Initialization or inference failure is exit 1, without
regex-only success or a partial clean manifest. There is no partial-NER opt-in.
`--no-ner` deliberately selects regex-only processing (inactive backend settings
are not applied); explicit CLI NER model/threshold controls require enabled NER.
A build without NER rejects
requested NER rather than pretending it ran. Error diagnostics omit private input
values and model/provider details. See [NER configuration](ner-backends.md).

A correct parser removes serialization-related mistakes, not all model false
positives. Use the synthetic [agent-trace gate](agent-trace-benchmark.md) to
measure recall and retained utility under an explicit policy before release.
