# Synthetic agent-trace precision/utility gate

`trx-dtg4`: offline, handwritten, **synthetic-only** pi-shaped traces. This is a
small challenge/regression corpus, not a representative sample of real sessions,
a parser fix, a confidentiality certification, or authorization to publish data.
No private trace was read. Production detector, format and configuration code
are not changed by this benchmark.

## Run

From the repository root:

```bash
just --justfile scripts/bench/trace.just test
just --justfile scripts/bench/trace.just check
just --justfile scripts/bench/trace.just bench --split all
just --justfile scripts/bench/trace.just gate
```

The **current corrected regex gate passes**: all 32 regex-scope occurrences are
removed, collateral bytes are zero and 104/104 approved benign occurrences survive.
The same strict gate remains red for the measured cached NER configurations.
The historical failing baseline is retained below, not accepted by weakening the
gate. Measurement without `--gate` exits zero if execution and validation succeed,
even when `gate_passed` is false. Controls cannot be gates.

Recipes use an isolated `/tmp/nym-benchmark-target`, overrideable with
`NYM_BENCH_TARGET_DIR`. Cargo builds are offline and regex-only. `NYM_BIN` selects
an existing binary without compiling; optional NER runs require a NER-enabled
binary. Mandatory Python accounting tests need only the standard library and
run through `uv`. No default model acquisition is performed.

```bash
NYM_BIN=target/debug/nym just --justfile scripts/bench/trace.just bench \
  --cached-token-model "$CACHED_TOKEN_MODEL" --split selection --threshold 0.5
NYM_BIN=target/debug/nym just --justfile scripts/bench/trace.just bench \
  --cached-token-model "$CACHED_TOKEN_MODEL" --split selection --threshold 0.5 --recall-first
NYM_BIN=target/debug/nym just --justfile scripts/bench/trace.just bench \
  --cached-token-model "$CACHED_TOKEN_MODEL" --split selection --threshold 0.9
NYM_BIN=target/debug/nym just --justfile scripts/bench/trace.just bench \
  --cached-token-model "$CACHED_TOKEN_MODEL" --split holdout --threshold 0.9 --gate
```

`CACHED_TOKEN_MODEL` must be an existing local directory containing tokenizer,
configuration and supported ONNX weight files. Repository IDs and missing files
are rejected before invocation. Token CPU argmax/recall-first are supported by
this harness. GLiNER/both are **not measured** here: no GLiNER cache was available,
and this harness does not acquire one. Model weights, tokenizer, configuration,
fixture and executable hashes are recorded; paths, matches and mappings are not.
Temporary config disables decision endpoints, uses the chosen local token model,
and isolates home/XDG state, working directory and inherited `NYM_*` settings.
HF/Transformers offline flags are set. Nothing edits global configuration.
Each case invokes native JSON detection and anonymization once each, loading
models per invocation. This is not a throughput benchmark.

## Corpus and policy

`scripts/bench/build_trace_fixture.py` deterministically expands handwritten
markers into `scripts/bench/fixtures/agent_traces.json`. Markers specify labels;
the detector/model does not create the gold. Independent validation checks exact
UTF-8 slices, valid paths, non-overlapping/nonduplicate annotations, scope
consistency, unique case IDs and valid parent/tool-call references. A test checks
the generated fixture equals the checked-in fixture. Annotation correctness is
hand-authored, not externally adjudicated; mechanical validation cannot establish
that a human chose the right sensitivity label.

There are four five-message traces, two `selection` and two `holdout`, containing
prose, thinking, Rust snippets, tool calls/arguments/results, public dependencies
and URLs, local paths/loopback addresses, timestamps, usage/config numbers,
booleans, floating costs and nulls. Fictional names, reserved-domain emails,
fictional accounts, example/fabricated credential strings, PINs, birth dates,
identity UUIDs, codenames and internal hostnames are planted. None is intended as
a working credential or real person. All planted occurrences have separate byte
offsets; approved literals have occurrence annotations, including benign `4096`
and an authentication PIN `4096` in the same trace. Unannotated surrounding text
is reviewed synthetic negative context, not an ignored precision region.

The **agent-trace-v1** policy uses all regex patterns at low confidence, except
`date,time,ipv4,ipv6,unix_path,social_url`. Those exclusions deliberately retain
build dates/times, loopback listeners, public system paths and dependency URLs.
This is not a suitable blanket policy for arbitrary infrastructure traces: real
network addresses and paths may be sensitive. UUID detection stays enabled for
identity metadata. Schema-specific own IDs, parent references, record timestamps,
tool-call IDs and tool-result references are excluded only in recognized trace
shapes. Tool names such as `bash`, type/role strings and unknown metadata are
scanned. There is no `**.id` or blanket metadata whitelist; an arbitrary metadata
ID or ID inside a text block is not exempt.

Detection and actual anonymization both run on the complete object using native
`--format json`, identical detection settings, explicit structural exclusions and
placeholder strategy. Native detection returns schema paths and leaf-local UTF-8
start/end offsets. The gate validates those offsets against decoded strings and
reconciles predictions against actual JSON output; it does not synthesize output
from detections and call that utility. The historical v1 matrix predates native
JSON offset output and instead detected each decoded leaf separately, then
reconciled against native whole-object anonymization.
It does not exercise automatic JSONL sniffing, streaming, file extensions or all
pi event variants; those belong to the structured-input tests. Sensitive numeric
scalars, object keys and binary payloads are outside this string-only policy;
scalar preservation is not evidence that arbitrary metadata is safe.

## Accounting and gate

- Reports include every gold, benign and predicted occurrence offset. `field`
  is the zero-based insertion-order ordinal in `scan_paths(case.trace)`; fixtures
  supply the corresponding schema paths. Raw-control ordinals reference the
  serialized field, with offsets independently mapped through JSON escaping.
- Coverage is the **union of UTF-8 bytes at each annotated position**. Duplicate
  findings cannot inflate recall. Strict full-byte, partial and non-whitespace
  coverage are separate. A person with only intervening whitespace surviving
  gets non-whitespace credit, **not** strict full-byte credit. Any leaked letter,
  combining mark or secret byte fails the relevant coverage requirement.
- Finding FPs have no gold overlap. Byte FPs also count collateral spill from a
  finding that touches gold; detection-count precision alone would hide this.
  Byte precision is covered gold bytes / all detected bytes (null if none).
  Scoring is label-agnostic; it is not category-classification accuracy.
- Actual-output reconciliation requires unchanged source chunks plus only a
  finite independently approved set of native placeholders at detected spans.
  Unknown tokens (including `<PIN4096>`), overlaps, partial replacements or any
  unexplained change fail closed. This proves source-byte removal under the
  placeholder contract, not semantic unlinkability of arbitrary pseudonyms.
- Approved benign literal preservation is **positional and reconciled against
  actual output**, not a document-wide or field-wide substring count. Removing
  a planted PIN must not penalize a different benign occurrence of `4096`.
  An unprovable output marks preservation unproven and fails the gate.
- The native output must retain the complete object/list shape, keys, scalar
  types/values and all declared structural IDs/references. Native Rust tests
  additionally assert the entire transformed document, not individual fields.

`--gate` requires zero errors, collateral FP bytes, output mismatches, structural
changes or benign loss, and complete transformed coverage for all in-scope gold.
Regex scope: emails, accounts, example/fabricated keys and sensitive identity
UUIDs. Adding a cached token model also gates person/PIN/birth-date coverage.
Only `person` permits a whitespace-only gap; other classes require every byte.
Known-sensitive codenames and internal identifiers remain reported `policy`
classes but are not automatically gated as universal detector capabilities.
They need explicit known-sensitive policy/review before sharing; even when a
model happens to catch one, that is not a general discovery guarantee.

## Current corrected measurement

`scripts/bench/reports/agent_trace_regex_corrected.json` records the mandatory
all-cases regex gate after `trx-qvav`. The unchanged fixture and unchanged strict
accounting give **32/32 full-byte transformed regex-scope occurrences, 0 FP
findings/bytes, 104/104 benign preserved, zero output mismatches and zero structural
changes**. The gate exits zero. Out-of-scope names/PINs/birth dates and policy
identifiers still have zero regex coverage and remain visible; this is not a
full-trace confidentiality pass.

`scripts/bench/reports/agent_trace_measurements_corrected.json` repeats the cached
matrix with native JSON detect+anon on the **same corrected binary**, comparing
regex and token configurations on identical selection/holdout cohorts. Executable
SHA-256 for both corrected files:

```text
0f0cfc62583deb30db55814e0137de27ce04464ea14f622a16dc55c47c3e9236
```

| Input/model | Split | Byte precision | FP findings / bytes | Benign preserved | Gate |
|---|---|---:|---:|---:|---|
| Native parsed regex | selection | 1.0000 | 0 / 0 | 52/52 | pass |
| Native parsed regex | holdout | 1.0000 | 0 / 0 | 52/52 | pass |
| Native token argmax 0.5 + regex | selection | 0.8305 | 10 / 109 | 42/52 | fail |
| Native token recall-first 0.5 + regex | selection | 0.8310 | 10 / 110 | 42/52 | fail |
| Native token argmax 0.9 + regex | selection | 0.8633 | 7 / 83 | 46/52 | fail |
| Native token argmax 0.9 + regex | holdout | 0.8612 | 7 / 83 | 46/52 | fail |
| Raw token argmax 0.9 + regex control | selection | 0.6256 | 38 / 289 | 42/52 | diagnostic |
| Native broad-regex policy control | all | 0.8297 | 12 / 164 | 92/104 | diagnostic |

Every completed corrected row has zero errors, output mismatches and structural
changes. Removing username-prefix collateral does **not** fix model utility:
parsed token rows still over-redact approved literals. Each token split again has
4/4 non-whitespace person coverage but 0/4 strict full-byte person coverage, and
2/2 full PIN and birth-date coverage. The same cached small/int8 checkpoint below
was used; GLiNER remains **unmeasured**, not a pass. No model is promoted.

## Historical pre-correction baseline and cached comparison

`scripts/bench/reports/agent_trace_measurements.json` retains the full value-free
reports, including offsets, **before** the contextual username-prefix correction
(`trx-qvav`). Measurements used one immutable snapshot of the parent's then-current
`target/debug/nym`, with the same fixture, parsed policy and binary for regex and
token rows. These rows are historical evidence, not post-correction measurements.
The invocation checkout was dirty; its Git SHA alone does not reconstruct that
binary. Executable SHA-256:

```text
2688f63b760cf075261d896163ff0f2df160f2f3336a2a8cafeb729e126db463
```

The only evaluated NER checkpoint was the cached public
`Wismut/nym-pii-multilingual-small/int8`, revision
`4348999cd3c2e20c49615e9af7c6bbb45b64cd85`, on CPU. Weight SHA-256:

```text
139006aea2cbd8e709d322f056232570de54661f624143be4893aaa387190286
```

| Input/model | Split | Byte precision | FP findings / bytes | Benign preserved |
|---|---|---:|---:|---:|
| Parsed regex | all | 0.9547 | 0 / 36 | 104/104 |
| Parsed token argmax 0.5 + regex | selection | 0.8079 | 10 / 127 | 42/52 |
| Parsed token recall-first 0.5 + regex | selection | 0.8087 | 10 / 128 | 42/52 |
| Parsed token argmax 0.9 + regex | selection | 0.8384 | 7 / 101 | 46/52 |
| Parsed token argmax 0.9 + regex | holdout | 0.8360 | 7 / 101 | 46/52 |
| Raw token argmax 0.9 + regex control | selection | 0.6114 | 38 / 307 | 42/52 |
| Parsed broad-regex policy control | all | 0.7998 | 12 / 200 | 92/104 |

All completed rows had zero execution errors, structure changes and output
mismatches. **No historical row passes the strict gate.** Regex removes all 32 regex-scope
occurrences; it misses all 8 names, 4 PINs, 4 birth dates and 8 policy-required
occurrences. Its 36 collateral bytes are four `username=` label prefixes consumed
by the old contextual matcher, not 36 leaked secret bytes. The
benchmark exposed this loss instead of relabeling those bytes as sensitive to
improve precision. The native regression test now expects value-only replacement,
retaining the username label; a corrected regex measurement is recorded separately.

Each parsed token selection row and the 0.9 holdout row covers all 4 person
occurrences' non-whitespace bytes but **0/4 strict full-byte person spans**;
spaces remain between separately redacted names. PINs and birth dates are 2/2
full-byte covered per split. At 0.9, the public dependency URL, prose buffer-size
`4096` and build timestamp are wrongly redacted in both cases; raw serialization
additionally wrongly redacts tool-name `bash`. Parsed traversal avoids that tool
name error here, not all model false positives. Broad regex damage is deliberate
policy behavior, not attributed to NER. Raising the threshold reduced observed
FPs but did not resolve utility failure; total covered gold bytes also decreased,
including partial policy-required codename detections. No larger/FP32 model was
promoted or measured.

0.9 was measured on the separate holdout after selection comparisons, without
further holdout tuning. Names/accounts/emails/PINs/dates/codenames differ across
splits, but the **templates and fixed secret/UUID controls are shared**. This is
a lexical holdout, not independent task/domain validation. Four templated traces
cannot estimate real-session precision or justify publication. Review annotation
and policy gaps, and expand independent scenario families before making any
practical de-identification quality claim.
