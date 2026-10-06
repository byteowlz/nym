# Local System-1 adjudication

`trx-pq2s.4`, research/prototype; not enabled for trace export. This safety
contract supersedes the older decision-layer description of model vetoes and
“calibrated” confidence. A typed probability distribution is not calibration.

## Implemented safety boundary

`src/engine/decision.rs` keeps existing public request/configuration entry points:

- Literal loopback only (`127.0.0.1`/`[::1]`); no environment proxy, redirects,
  raw request/reply logging, or remote fallback. A loopback address alone does
  not prove a listener does not forward: the operator must own its runtime.
- Both backends send bounded candidate context, never the full document.
  Context counts Unicode scalars; candidate offsets remain original UTF-8 bytes.
- Complete answer count/unique IDs required. Mixed positional/indexed answers,
  duplicate IDs/map keys, wrong types, invalid scores/verdicts, missing/partial
  answers and malformed envelopes fail atomically. Transport/timeout errors
  cannot return earlier batches' decisions. Error messages omit model values.
- Budget limits requests, not coverage: unreviewed candidates remain present.
- All `detector`/unknown provenance and user-sensitive matches stay `Redact`.
  The current merged `PiiMatch` does not distinguish regex from NER, so this is
  deliberately conservative. Explicit `ner`/`entropy` candidates are advisory.
- Model `keep` becomes unresolved `Flag`, irrespective of claimed confidence.
  A `Flag` or error means redact/retain/block export, **not raw pass-through**.
  Existing `decide` reports only; it does not rewrite or certify a document.

## Measured synthetic trials, 2026-10-06

Owned Mac, existing cached weights, sequential requests, offline runtimes,
loopback-only listeners. No private input, downloads, global configuration
changes, or busy-job preemption. A remote owned GPU was busy and left untouched.

20 handwritten cases (10 private, 10 benign), two passes/model. Same-value
counter/PIN and build-date/DOB pairs, negation, public/private URLs, code
credentials/placeholders, checksums, explicit sensitive terms, prompt injection,
and Unicode offsets. **Candidates are supplied by the fixture**: this measures
conditional adjudication, not discovery or whole-document confidentiality.

| Lane | Valid replies | First-pass outcomes | p50 / p95 request | Deduplicated cache files | Observed host RSS max |
|---|---:|---|---:|---:|---:|
| Kev-0.5B, native Choice, Torch fp32/MPS | 40/40 | 7 redact, 13 flag, 0 keep | 39.6 / 43.4 ms | 1.053 GB incl. base | 1.145 GB |
| Qwen2.5-1.5B-Instruct 4-bit, MLX | 0/40 | Strict JSON contract rejected all | 267.1 / 407.4 ms | 0.880 GB | 1.201 GB |

Kev retained/flagged all 10 private candidates but preserved **0/10 benign**;
this equals the redact-all candidate baseline, not an improvement. The instruct
lane has **no valid quality estimate**, not “100% model recall.” Its failures
were JSON decoding/format failures; diagnostic replies included Markdown fences.
Operational failure retention is reported separately from model correctness.
Neither result authorizes an automatic benign decision.

An actual regex-only `nym` all-rules/default-confidence control fully covered **1/10
private** candidates and left **9/10 benign** untouched (112.1 ms median including
process startup). This broad control is not the trace-specific policy benchmark.
The independently measured token-NER trace baseline in
[agent-trace-benchmark.md](agent-trace-benchmark.md) uses a different corpus;
its figures must not be combined into a model ranking here.

Repeated-request medians: Kev 40.2 ms, instruct 262.3 ms. Cache hit rate, cold
model-load time and device/unified-memory residency are **unmeasured**; RSS is
not accelerator allocation. Both were resident during trial; this is not a
swap/eviction benchmark. Cached Kev adapter revision:
`edf1dc6d7f8d983c0adfd251e80a686e5539fc61`; Qwen base revision:
`060db6499f32faf8b98477b0a26969ef7d8b9987`; MLX revision:
`8b403126fc14f14cfc99bb4cfa72ecbc129ea677`.

## Reproduce and safety proof

No harness input-text option exists. An endpoint requires an explicit owned
loopback attestation; the harness never launches/downloads a model. Runtimes
must already serve cached weights. For native Choice:

```bash
uv run --offline --no-project scripts/bench/system1_bench.py \
  --endpoint http://127.0.0.1:18769/v1/systemone --owned-loopback \
  --backend systemone --model kev-0.5b --repeats 2 \
  --output /tmp/nym-system1.json
# Optional --server-pid PID, --artifact CACHE_DIR (repeatable), --nym BINARY.
# For instruct: --backend chat and /v1/chat/completions.
uv run --offline --no-project -m unittest discover -s scripts/bench -p 'test_system1*.py'
CARGO_TARGET_DIR=/tmp/nym-system1-target cargo test --offline \
  --bin nym --no-default-features --features decision system1_
```

The focused Rust suite passes 12 tests; the harness suite passes 8 tests.
Local deterministic HTTP stubs prove failure handling, candidate completeness,
protected-match retention, budget coverage and stable offsets. They do **not**
prove model judgment. Three original regressions were observed failing before
the fix: detector veto, truncated coverage and empty Unicode context.

## Integration proposal and next comparison

Keep the judgment runtime replaceable; nym owns deterministic identity, coverage,
transport authority and export policy. Proposed small seam before NER merging:

```text
Candidate { id: (field_ordinal, start_byte, end_byte), origin, span, context }
Advisory { id, recommendation, option_scores?, model_revision, schema_revision }
Review { id, retained | policy_keep | unresolved, policy_revision }
```

`origin` must be trusted typed provenance (`regex`, `user_sensitive`, `ner`,
`entropy`), not inferred from a model label. Regex/user-sensitive always retain.
Trace policy alone can accept a **complete soft candidate**; missing candidates,
backend failure or invalid spans block export/retain all. Audit counts include
reviewed/unreviewed, abstentions and transport failures, with no values. This
belongs before merging in the parent-owned trace-policy path; not in this patch.

Compare a **trained bidirectional encoder privacy head** on candidate + left/right
context against a small instruct model and native Kev using identical family
holdouts. Cached encoder backbones exist, but no verified contextual privacy head
was found or trained here: encoder accuracy/latency are **unmeasured**, not assumed
from parameter count. A pretrained embedding model is not such a classifier.
Native Kev-4B and better schema-constrained instruct models remain unmeasured.

Fit temperature/isotonic calibration on a separate calibration split, then choose
selective benign acceptance by held-out false-keep risk and coverage, not a guessed
0.9 confidence. Report per-family sensitive recall, benign utility, Brier/ECE,
latency, actual device residency, startup and cache behavior. The exploratory
threshold table in this harness is **not deployment authorization**; there are
zero accepted native keeps here. Even 10 independent zero-error accepts would
only bound false-keep risk below about 25.9% at one-sided 95% confidence. Preserve
multilingual positive replay and evaluate the exported/quantized encoder artifact,
not just its training checkpoint. Keep advisory-only until that evidence exists.
