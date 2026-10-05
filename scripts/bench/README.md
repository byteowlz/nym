# nym synthetic session-fixture benchmark

Reproducible, offline recall / false-positive / utility-preservation harness
for nym on a versioned, independently labeled synthetic coding-agent session
challenge set. Emits **only safe aggregate results** — no matched PII values,
no source paths, no reversible mappings — so the output is suitable for a
public run manifest or CI.

## Files

- `fixtures/challenge.json` — the labeled challenge set (versioned).
- `run_bench.py` — the runner (invokes the nym binary, measures, reports).
- `bench.sh` — wrapper that builds a release binary and records pinned versions.

## Usage

```bash
# Offline regex-only (default; no model download, no network egress)
just bench

# With a regression gate on per-class recall (in-scope classes only)
just bench-gate --fail-on-recall 0.9

# Machine-readable aggregate output
NYM_BIN=target/release/nym python3 scripts/bench/run_bench.py --json
```

Every run records the fixture version, nym binary version, git rev, and the
mode so results are reproducible and attributable.

## Model-backed modes (optional)

The default recipe is regex-only and requires no downloads. NER and
decision-model recall are out of scope for the offline recipe because they need
a locally cached model (NER) or an external endpoint (decision). Those recipes
must be enabled explicitly and **declare** their requirements:

- `NYM_NER_CONFIG` — a config file that enables NER, run with `... --ner`.
  Requires a locally cached model; do not install one into git.
- Decision-backed review requires an OpenAI-compatible endpoint, which needs
  explicit egress approval and is **not** exercised here.

## What is measured

Per-class recall (value-based span coverage, so partial matches count),
false positives (findings not overlapping a gold span), and benign-literal
preservation (task-relevant content that should survive untouched).

Classes that regex patterns cannot detect (`person`, `organization`,
`internal_hostname`, `codename`, `unlabeled_high_entropy`) are reported but
flagged `out_of_regex_scope`, and are **excluded** from the recall gate in
regex-only mode. Do not interpret a low recall there as a regex failure.

## Scope limits

- Synthetic only: no private trace, credential, or real user data is checked in.
- Reporting denominators, false negatives, and class scope are explicit.
- This harness does **not** establish a universal safe threshold; treat
  per-class recall as evidence within this fixture set, not a guarantee.
- Human-review protocol for proprietary content not reducible to PII spans, and
  rights/consent and weight-extraction testing, remain separate checks.

## Regression gate

`--fail-on-recall X` exits nonzero when an in-scope class recall falls below
`X`; `--fail-on-benign` additionally fails when a benign literal is flagged.
The default gate fails on seeded misses (in-scope gold spans undetected) and on
fixture corruption (parse errors, malformed fixture).