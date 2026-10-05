# Synthetic session-fixture benchmark

Offline, independently labeled synthetic fixtures. Reports contain only
aggregates and hashes: no matched values, source paths, or reversible mappings.

## Run

```bash
just bench
just bench-gate 0.9
bash scripts/bench/bench.sh --json --fail-on-benign
uv run --no-project -m unittest discover -s scripts/bench -v
```

The wrapper runs Cargo's freshness check with `--no-default-features` in the
isolated `target/bench/` build directory, then invokes the runner through `uv`.
`NYM_BIN` overrides the binary and skips building; its hash identifies the
actual executable rather than claiming it came from the current source.

Direct invocation (after building):

```bash
NYM_BIN=target/bench/release/nym uv run --no-project scripts/bench/run_bench.py --json
```

## Validation and metrics

- Empty or absent gold/benign literals, conflicting labels, malformed fixtures,
  and invalid finding byte offsets fail the run before misleading metrics can
  be accepted. Fixture validation occurs before invoking nym.
- Gold values label every occurrence in the input. Recall uses the union of
  overlapping UTF-8 byte ranges at their actual positions, never substring
  similarity. Duplicate findings cannot inflate coverage.
- `recall` counts **fully covered** gold occurrences only. `partial_spans` and
  `partial_recall` count occurrences with some but incomplete byte coverage.
- False positives are findings with no positional overlap with any gold span,
  counted by detector category. A distinct benign annotation counts once per
  fixture, regardless of repeated occurrences or overlapping findings.
- `benign_literals_flagged` measures detection overlap. `benign_preservation`
  independently measures actual `anon` output: every occurrence of each benign
  literal must survive. Anonymization uses deterministic placeholders with the
  same format, confidence, ruleset, and NER settings as detection. Literal counts
  measure preservation, not positional identity or universal redaction safety.
- Every run reads the invocation checkout's full Git revision and dirty status,
  hashes the executed binary and fixture file, and includes ordered SHA-256
  hashes of canonical fixture objects. No shared `/tmp` provenance is consulted.
  Git revision describes the invocation checkout, not an override binary's
  build source; dirty/untracked source is not reproducible from the SHA alone.

## Gates and scope

Execution/validation errors always fail. `--fail-on-recall X` gates unrounded
**full byte-span** recall for in-scope occurrences, not partial matches.
`--fail-on-benign` additionally fails on detection overlap or actual literal
loss. Neither `just bench` nor the recall gate implicitly gates benign utility.

Contextual classes (`person`, `organization`, `internal_hostname`, `codename`,
`unlabeled_high_entropy`) are excluded from regex-only recall gating, but remain
reported. Per-fixture scope exclusions never hide another fixture's in-scope
miss for the same class. No universal anonymization-safety claim follows.

Optional `--ner` requires an explicit `NYM_BIN` with NER support and a locally
cached model; `NYM_NER_CONFIG` supplies its configuration. No model download or
external decision endpoint is exercised by the default recipe.

## Change log

- Validate annotations; separate full/partial byte coverage and detection vs
  actual anonymization preservation; capture fresh checkout/binary/fixture
  provenance; rebuild in an isolated directory using `uv`.
- Correct absent benign labels and add account/handle recall fixtures. Bare
  numeric/underscore code tokens no longer imply usernames: conventional
  `userNNN` identifiers and explicitly labeled account fields remain supported.
  Labeled username matching redacts the entire field, including its label;
  arbitrary unlabeled names such as `john_doe` now require account context.
