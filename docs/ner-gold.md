# Complete-gold model adaptation

Reducing unnecessary redactions must not regress detection of real sensitive values. Vocabulary coverage and model-only quality are separate measurements: a literal rule catching a model miss cannot hide a model regression.

## Data contract

Collect complete occurrence-level annotations, including entities the current detector missed and explicitly reviewed negatives. Each JSONL row uses `nym.ner.gold.v1`:

```json
{
  "schema": "nym.ner.gold.v1",
  "source_id": "document-a",
  "unit_id": "paragraph-1",
  "text": "original unmodified decoded text",
  "text_sha256": "SHA256_OF_UTF8_TEXT",
  "annotation_complete": true,
  "entities": [{"start": 0, "end": 8, "label": "PERSON"}],
  "masked_spans": [{"start": 9, "end": 19}]
}
```

This illustrates the shape, not a ground-truth labelled example. Offsets are Unicode codepoints, end-exclusive; native NYM findings use UTF-8 bytes and require explicit conversion. No normalization is allowed before checksumming or annotation. Labels are declared entity types, not private values. Uncertainty belongs in `masked_spans`, never an O label or an `UNCERTAIN` entity class. Known and unknown spans cannot overlap.

A reviewed negative has `annotation_complete=true`, empty entities and empty masks. A partly reviewed or whole-row needs-context example has `annotation_complete=false`; preparation excludes it completely. Vocabulary approvals and findings-only votes are not this schema and must not be imported as complete gold.

## Prepare a frozen experiment

Before extraction/annotation, freeze source identities and a JSON map assigning each source to `train`, `selection` or `holdout`. Declare the complete taxonomy independently of the holdout contents in a JSON string array. Do not partition with changing newest/skip queries.

```bash
uv run --no-project python scripts/prepare_ner_gold.py \
  --input reviewed.jsonl --source-splits frozen-splits.json \
  --label-types taxonomy.json --output private-gold-bundle.json
```

Preparation validates original checksums, exact schemas/types, range bounds, nonoverlap, taxonomy, duplicate unit identities, source-disjoint partitions and exact text-copy leakage. All three retained partitions must be nonempty; optimizer data must include known sensitive positives and fully reviewed negatives. Incomplete rows are counted as exclusions. Output is one checksummed, owner-only, atomic no-clobber bundle; no private path/value is reflected in CLI preparation errors.

Exact-copy rejection is not near-duplicate/family disjointness. Audit related sessions, copied templates and near duplicates separately before claiming independent evaluation. Checksums prove consistency, not annotation truth or authorship. Calibrate a small pilot before expanding the dataset.

## Train without turning unknowns into negatives

```bash
uv run --no-project python scripts/train_ner.py \
  --gold-bundle private-gold-bundle.json --base-model LOCAL_TRAINABLE_CHECKPOINT \
  --output-dir PRIVATE_OUTPUT --max-length 512 --no-bf16 --dry-run
```

The dry run is counts-only: no private token dump. Gold mode validates partitions before loading tokenizer/model assets and rejects weak-negative promotion or unmasked distillation. A token with **any overlap** with an uncertain region is loss-masked (`-100`), even if its start lies outside that region. Ambiguous entity-boundary-crossing tokens are masked too. Gold units exceeding the token limit fail rather than silently truncate; split and re-annotate them. Fully masked optimizer/selection units cannot contribute misleading O loss. A dataset with no supervised sensitive tokens is refused.

Gold mode preserves matching label IDs by default. For an intentional architecture/taxonomy change, `--allow-head-reset` explicitly authorizes a new classifier; it never overrides `--frozen-label-config`. Frozen continuation still requires every learned tensor and the exact BIO map. This does not certify encoder/export fidelity for a new model family.

Remove `--dry-run` only after annotation calibration and a predeclared bounded local protocol. Gold bundles use selection for checkpoint choice; bundled holdout text is not automatically inferred after training. The trainer's selection F1 is advisory, not a privacy eligibility result. Its masked token metrics are not full-value packaged-pipeline recall. Legacy synthetic training options remain available; unknown label types now fail rather than silently becoming O.

## Independent model-only selection gate

```bash
uv run --no-project python scripts/eval_ner_gold.py \
  --gold-bundle private-gold-bundle.json --binary IMMUTABLE_NYM_BINARY \
  --baseline-model LOCAL_BASELINE_ONNX --candidate-model LOCAL_CANDIDATE_ONNX \
  --output private-model-only-report.json
```

This CPU-only runner pins binary/model hashes and threshold 0.5, disables every registered regex and all decision/lexicon policies in isolated configuration, and requires an empty no-NER control on the same inputs. Native NER pattern names can overlap regex names; prefixes are not provenance. It evaluates **selection only**, checks original UTF-8 slices, and reconciles actual anonymized output against every proven range; unsupported accounting fails, never drops a unit. Unknown regions are excluded from collateral/negative supervision metrics and reported separately.

The predeclared strict gate requires no per-class full-removal regression, complete full-value removal for every supported sensitive class, reduced collateral bytes without more false findings or fewer preserved negatives. Unsupported taxonomy classes are explicitly unmeasured, never reported as passed. Exit 1 means incomplete/operational failure; exit 2 means completed policy blockage; exit 0 only makes a selection candidate eligible for further gates. None authorizes promotion or reserved-validation use automatically. Counts-only reports are private, atomic and no-clobber. This runner does not replace source/near-duplicate auditing, natural annotation quality, combined-pipeline/quantization tests or strict restoration gates.

## Promotion is separate

1. Freeze the original baseline, checkpoint-selection rule, per-class full-value recall floors and utility goals before experiments.
2. Evaluate **model-only** at threshold 0.5 on independent sensitive positives, fully reviewed negatives and technical/sensitive contrasts. Separately report lexicon-plus-model coverage.
3. Require contextual false-positive improvement without any supported sensitive-class recall regression. Classes with no support are unmeasured, not passed. Test actual credentials/PIN/DOB/names and complete values, not only BIO F1.
4. Check FP32 numerical/decision parity and actual packaged NYM detection, redaction and strict restoration. Failure/incomplete inference is not a clean result.
5. Only then quantize and repeat fidelity, full-value privacy gates and utility, with actual package size, RSS and target-device timing.
6. Only an eligible selection winner reaches frozen reserved validation and unchanged strict deployment gates. Never switch checkpoints after validation failure, repair with thresholds or activate automatically.

Clinical33 remains a first adaptation experiment, not a qualified stock replacement. Tooling and synthetic tests do not mean a trained model meets these gates. Private annotations and weights stay outside git and external model services. No training or activation starts from either review UI.
