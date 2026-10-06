# Agent-trace privacy policy

The default detector is unchanged. Opt in explicitly:

```bash
nym detect --ner --profile agent-trace --public-host docs.rs trace.jsonl --summary-json
nym anon --ner --profile agent-trace --sensitive-terms-file "$HOME/.config/nym/sensitive.txt" trace.jsonl
```

The profile retains unknown contexts. It only filters NER candidates with supported technical context or explicitly trusted public-reference hosts; it never skips entire code or metadata blocks. Credential regex findings and sensitive-list matches cannot be vetoed. A threshold change is not a repair for contextual model errors; the NER default remains 0.5.

## Literal lists

Configure `[trace_policy]` in `config.toml` (see [example](../examples/config.toml)), or use repeatable `--sensitive-terms-file`, `--benign-terms-file`, and `--public-host`. CLI lists replace corresponding configured lists. Files contain one literal per nonblank line; nonblank lines preserve whitespace and have no comment syntax. Duplicate terms are deduplicated deterministically. Paths expand `~` and environment variables. Keep private lists outside git.

Sensitive terms work independently of the profile and win over benign terms. Benign terms require the profile, cover a whole candidate, and cannot override protected classes, private context, or regex findings. `--term-boundary word` uses Unicode word characters, including combining marks and underscores; `substring` also matches inside paths/identifiers. `--term-case-sensitive false` uses Unicode simple case folding, not normalization. Defaults are word boundaries and case sensitivity. `--profile default` disables heuristics, not configured sensitive lists.

Audit summaries expose aggregate candidate/suppression/list counts, not terms or snippets. Counts precede overlap merging and are not a precision estimate.

## Restoration after external editing

`deanon` verifies restoration by default. It authorizes only complete, exact aliases from the key and verifies the original decoded input outside proven restored spans. Inserted originals are never rescanned or cascaded. Recognizable case variants, components, ambiguous components, and unchanged JSON keys block publication with counts-only diagnostics. Components are evidence for review, not authorization to guess an original name. Short aliases inside words are not restored.

Text, JSON, JSONL and supported office output is staged before publication. A later malformed record or verification failure leaves an existing destination unchanged and emits no staged stdout. `--no-verify-restore` explicitly selects legacy exact/component restoration without the residual guarantee.

Office verification operates on original decoded paragraphs, including split runs, entities and CDATA, and checks unchanged XML-family surfaces and archive entry names. DTDs, custom references, malformed XML and archives lacking a recognized document part fail closed. Embedded opaque images/binary objects and attachments with unrecognized extensions are outside XML verification coverage.

Recognizable residual checks cannot prove arbitrary paraphrase reversible. External tools must preserve full aliases; human review is necessary when they rewrite names or other pseudonyms beyond recognition.

## Evaluation and model status

- [Trace benchmark](agent-trace-benchmark.md): the original strict utility gate and independent family holdout remain unchanged.
- `scripts/datagen/agent_trace_corpus.py`: exact character spans (independently converted to UTF-8 bytes for deployed scoring), full negative supervision, same-value sensitive/technical contrasts, disjoint scenario families and frozen checkpoint label IDs. This is synthetic training data, not real-distribution quality evidence.
- `scripts/bench/local_trace_eval.py`: explicit local text selectors and occurrence-safe canary scoring; stratified human precision review with uncertainty bounds. Raw inputs, bundles and review exports stay outside all git working trees, with private file permissions. Do not upload them. A real precision estimate requires completed human reviews, not merely exported samples.
- `scripts/recover_small_checkpoint.py`: lossless recovery of the published small model's learned FP32 tensors into a trainable checkpoint; no random head replacement. Recovery parity and finite gradients do not establish improved weights.
- [Local System-1 study](local-system1.md): measured small-model trials did not establish safe benign utility improvement. Recommendations remain advisory; malformed/partial/timeout responses fail closed and hard findings remain protected.

Training, exported-artifact parity, sensitive recall, technical utility and real-session evaluation are separate proof obligations. No candidate should be promoted merely because tooling or synthetic safety tests pass. Models are not published by these workflows.

### Measured continuation: not promoted

A bounded three-epoch CPU continuation preserved the learned head and frozen labels. On the new synthetic family holdout, benign preservation improved from 16/96 to 49/96, but complete PIN recall fell from 24/24 to 22/24; DOB remained 48/48. Per-language synthetic controls also regressed in Russian and Japanese. The candidate was fixed using selection data, not switched after inspecting holdout.

FP32 export passed 605-case Torch parity with no argmax or 0.5 decision differences. INT8 export failed parity and further reduced deployed holdout PIN recall to 16/24. All original strict deployed selection/holdout gates failed: the candidate preserved 50/52 benign occurrences, with two false-positive findings and 56 collateral bytes. Neither artifact was promoted or published.

The independent deployment runner also rejected nested detector findings: 16 holdout and 59 public-control output reconciliations failed across the original and both candidate artifacts. Bounded full-value counterexamples showed exact outer-span anonymization with nested findings, not unknown placeholders or proven CLI corruption. The strict oracle and historical reports were not relaxed; these runs remain failed, not clean output proofs. Independent holdout had no person support; historical real OOD/final2 and real-session precision remain unmeasured.
