# PII training-data generator

Synthesize labeled PII token-classification data with **exact, noise-free labels**
for training your own nym token-classification model (BERT/DeBERTa).

## How it works

1. **Templates** with `[LABEL]` placeholders come from a **local LLM** (any
   OpenAI-compatible endpoint — LM Studio, Ollama, llama.cpp) and/or the built-in
   [seed bank](seeds.py). The LLM only supplies linguistic diversity; it never
   handles data or offsets.
2. **Faker fills** each placeholder with a realistic, format-correct, locale-aware
   value ([labels.py](labels.py)), building the string incrementally so every
   value's character span is recorded exactly — no search-and-replace, no offset
   drift.
3. **Negatives** (PII-free text) are mixed in to curb false positives.
4. Output is **char-offset JSONL** (`{"text", "entities":[{start,end,label}]}`),
   split into train/val/test, with a label-distribution report. Optionally also
   emits **HF BIO** token/tag pairs ready for `AutoModelForTokenClassification`.

> **Why not DSPy?** The correctness-critical work (offsets, BIO alignment, label
> validity) is deterministic Python, and "template diversity" has no clean metric
> to optimize — so DSPy would add a framework without touching the hard part. A
> thin structured-prompt call to a local model is the right amount of machinery.
> DSPy could later optimize the template prompt against a diversity/validity
> reward, but it isn't needed here.

## Usage

```bash
# Offline — seeds only (good for a smoke test / no GPU):
uv run --with faker scripts/datagen/generate.py --no-llm -n 500 -o data/pii.jsonl

# With a local model (LM Studio default port shown; Ollama = :11434):
uv run --with faker --with openai scripts/datagen/generate.py \
    -n 20000 -o data/pii.jsonl \
    --base-url http://localhost:1234/v1 --model your-local-model \
    --locales en_US,en_GB,de_DE,fr_FR,es_ES --fills-per-template 10

# Also emit BIO for training:
uv run --with faker --with transformers scripts/datagen/generate.py \
    -n 20000 -o data/pii.jsonl --to-bio bert-base-cased
```

Key flags: `--num`, `--fills-per-template`, `--locales`, `--neg-ratio`,
`--split`, `--seed` (reproducible), `--to-bio <tokenizer>`.

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
