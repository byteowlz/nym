# PII training-data generator

Synthesize labeled PII token-classification data with **exact, noise-free labels**
for training your own nym token-classification model (BERT/DeBERTa).

## How it works

1. **Rubric sweep** ([rubric.py](rubric.py)) — enumerate `(language × topic ×
   style)` cells across ~23 languages, 20 domains, 8 registers. For each cell a
   **local LLM** (any OpenAI-compatible endpoint — LM Studio, Ollama, vLLM) writes
   `[LABEL]`-placeholder templates *in that language and register*. This gives
   systematic breadth instead of whatever the model gravitates to. A [seed
   bank](seeds.py) works offline (`--no-llm`). The LLM only supplies prose; it
   never handles data or offsets.
2. **Faker fills** each placeholder with a realistic, format-correct value in the
   cell's **matching locale** ([labels.py](labels.py)) — a Korean template gets
   Korean names, a German one German IBANs. The string is built as segments so
   every value's character span is recorded exactly.
3. **Noise** ([noise.py](noise.py)) — a configurable fraction of examples is
   corrupted with **label-preserving** OCR confusions (`o→0`, `rn→m`, …), keyboard
   typos, spacing glitches, and case flips. Corruption is applied per-segment and
   offsets recomputed, so spans stay exact even when a name becomes `MMiren0` or
   an address is smudged — teaching the model to tag messy real-world text.
4. **Negatives** (PII-free text) curb false positives.
5. Output is **char-offset JSONL** (`{"text", "entities":[{start,end,label}]}`),
   split train/val/test, with a label-distribution report. Optionally also emits
   **HF BIO** token/tag pairs ready for `AutoModelForTokenClassification` (use a
   *multilingual* tokenizer for the non-Latin languages).

Key flags: `--cells` / `--per-cell` (rubric breadth), `--noise-ratio` /
`--noise-level` (light|medium|heavy), `--fills-per-template`, `--neg-ratio`,
`--split`, `--seed`, `--to-bio <tokenizer>`, `--dump-templates`.

> **Why not DSPy?** The correctness-critical work (offsets, BIO alignment, label
> validity) is deterministic Python, and "template diversity" has no clean metric
> to optimize — so DSPy would add a framework without touching the hard part. A
> thin structured-prompt call to a local model is the right amount of machinery.
> DSPy could later optimize the template prompt against a diversity/validity
> reward, but it isn't needed here.

## Usage

```bash
# Offline — seeds only (smoke test / no GPU), with 20% noisy examples:
uv run --with faker scripts/datagen/generate.py --no-llm -n 500 -o data/pii.jsonl

# Full multilingual rubric sweep on a local model, 20% noisy + BIO:
uv run --with faker --with openai --with transformers scripts/datagen/generate.py \
    -n 20000 -o data/pii.jsonl \
    --base-url http://hp-z8:8080/v1 --model step-3.7-flash \
    --cells 300 --per-cell 12 --noise-ratio 0.2 --noise-level medium \
    --to-bio bert-base-multilingual-cased
```

`--cells` × `--per-cell` sets template breadth (300 cells × 12 ≈ 3,600 templates
spanning many languages/domains); each template is then filled `--fills-per-template`
times. `--seed` makes everything reproducible.

## Labels

The placeholder vocabulary (`labels.ALLOWED_LABELS`) matches nym's
`label_to_pattern` mapping, so a model trained on this data integrates cleanly.
Add a label by adding a Faker generator in [labels.py](labels.py) (and, for nice
`<LABEL>` replacement in nym, an arm in `src/engine/ner_token.rs::label_to_pattern`).

## Output format

```json
{"text": "Passport Y94316756 issued to Gary Fisher, Ukraine, expires 11/03/2008.",
 "entities": [{"start": 9, "end": 18, "label": "PASSPORT"},
              {"start": 29, "end": 33, "label": "GIVEN_NAME"},
              {"start": 34, "end": 40, "label": "SURNAME"},
              {"start": 42, "end": 49, "label": "COUNTRY"},
              {"start": 59, "end": 69, "label": "DATE"}]}
```

Offsets are **character** indices (matches HF fast-tokenizer `offset_mapping` for
BERT/DeBERTa, so BIO conversion is exact including non-ASCII). This is also the
shape `nym bench` reads, so you can evaluate a trained model with
`nym bench data/pii.test.jsonl --ignore-labels`.
