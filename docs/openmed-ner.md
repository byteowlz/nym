# OpenMed NER backend

nym supports **two NER backends in parallel**:

| Backend  | Model kind | Crate / runtime | Labels |
|----------|------------|-----------------|--------|
| `gliner` (default) | GLiNER zero-shot **span** model | `gline-rs` | Arbitrary, zero-shot (you supply labels) |
| `openmed` | [OpenMed](https://github.com/maziyarpanahi/openmed) DeBERTa-v2 **token classification** | `ort` (ONNX Runtime) directly | Fixed 106-label PII taxonomy (BIO) |

The OpenMed models are fine-tuned specifically for clinical/HIPAA PII and recognise
entity types regex cannot (names, cities, dates of birth, medical record numbers,
…). GLiNER stays useful for ad-hoc, zero-shot labels. You can run either alone or
**both at once** — results are merged and de-duplicated by span overlap.

## 1. Convert an OpenMed model to ONNX

OpenMed publishes PyTorch checkpoints; nym loads ONNX. Convert one with:

```bash
# Defaults to the small 44M PII model -> models/<name>-onnx/
scripts/convert_openmed_onnx.sh

# Or pick a model + output dir:
scripts/convert_openmed_onnx.sh OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1 models/openmed-base
```

This produces `model.onnx` (fp32), `tokenizer.json`, and `config.json` in the
output directory. Requires [`uv`](https://docs.astral.sh/uv/); all Python deps run
in an ephemeral environment. Set `NYM_QUANTIZE=1` to additionally emit
`model_int8.onnx` (preferred by the backend when present) — see the quantization
caveat below; it is reliable only for the small model.

Recommended models (all `DebertaV2ForTokenClassification`, 106 BIO labels):

| Model | Size | Notes |
|-------|------|-------|
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` | small | fastest, good first choice |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1` | base | balanced |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1` | large | flagship accuracy |

## 2. Configure nym

```toml
[ner]
enabled = true
backend = "openmed"          # "gliner" (default) | "openmed" | "both"
# openmed_model accepts EITHER a local directory OR a HuggingFace repo id.
# A repo id (optionally with a subfolder) is downloaded + cached automatically:
openmed_model = "Wismut/openmed-onnx/small"   # also: /base, /large
# ...or a local converted dir:
# openmed_model = "/abs/path/to/models/OpenMed-PII-SuperClinical-Small-44M-v1-onnx"
threshold = 0.5

# For backend = "both", also set the GLiNER repo (defaults shown):
# model = "onnx-community/gliner_multi-v2.1"
```

`openmed_model` is resolved as a **local directory if one exists at that path**,
otherwise as a **HuggingFace repo id** — optionally with a subfolder
(`org/name/subdir`) so several models can share one repo — fetched via `hf-hub`
into the HF cache (`HF_HOME`, or `[ner] cache_dir`). So once the ONNX model is
published to the Hub, users need no manual conversion. The pre-converted models
used in this doc live at [`Wismut/openmed-onnx`](https://huggingface.co/Wismut/openmed-onnx)
(`/small`, `/base`, `/large`). See [Publishing models](#publishing-models).

Then:

```bash
nym detect input.txt --config nym.toml
nym anon   input.txt --config nym.toml -o out.txt
```

## How it works

`src/engine/ner_openmed.rs` implements the backend:

1. Tokenize with the HF `tokenizers` crate (offsets enabled).
2. Feed `input_ids` + `attention_mask` to the ONNX session (`ort`).
3. Argmax + softmax over the per-token 106-way logits.
4. BIO-decode contiguous tokens into entity spans.
5. Map sub-word offsets back to byte offsets and emit `PiiMatch`es, mapping each
   OpenMed label to nym's canonical pattern names where they overlap (so existing
   placeholder/fake replacement applies).

Long inputs are chunked by words with overlap, mirroring the GLiNER backend.

## Publishing models

OpenMed does **not** publish ONNX on the Hub (only `safetensors` + `spm.model`),
so the converted ONNX has to be hosted somewhere for auto-download to work. Each
model lives in a folder (repo root, or a subfolder) containing:

```
config.json
tokenizer.json
model.onnx            # fp32 (required)
model_int8.onnx       # optional; preferred when present (small model only — see
                      #   the quantization caveat, it breaks base/large)
```

That is exactly the layout `scripts/convert_openmed_onnx.sh` produces. The
reference repo [`Wismut/openmed-onnx`](https://huggingface.co/Wismut/openmed-onnx)
hosts all three sizes as subfolders (`small/`, `base/`, `large/`) under one
Apache-2.0 repo with OpenMed attribution. Recreate it with:

```bash
hf auth login                              # once
scripts/convert_openmed_onnx.sh OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1 models/small
scripts/convert_openmed_onnx.sh OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1  models/base
scripts/convert_openmed_onnx.sh OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1 models/large
hf upload <org>/openmed-onnx models/small small --repo-type model
hf upload <org>/openmed-onnx models/base  base  --repo-type model
hf upload <org>/openmed-onnx models/large large --repo-type model
```

Users then set `openmed_model = "<org>/openmed-onnx/small"` and nym downloads +
caches it on first run — no local conversion needed. OpenMed is Apache-2.0, so
redistributing the converted weights is fine; **keep the license and OpenMed
attribution** in the repo (the reference repo ships both).

## Benchmarks

Span-level detection on **1000 examples** of `ai4privacy/pii-masking-300k`, using
label-agnostic matching (`--ignore-labels` — counts any span overlap as a hit, so
backends with different label taxonomies compare fairly). Threshold 0.5, default
high-confidence regex patterns. Each NER row = regex + that backend (regex always
runs).

| Backend                | Precision | Recall | F1    |
|------------------------|-----------|--------|-------|
| regex only             | 79.2%     | 27.8%  | 41.2% |
| GLiNER                 | 66.8%     | 63.3%  | 65.0% |
| OpenMed small (int8)   | 71.4%     | 74.6%  | 73.0% |
| OpenMed base (fp32)    | 69.6%     | 79.4%  | 74.1% |
| OpenMed large (fp32)   | 70.9%     | 82.3%  | **76.1%** |
| both (small + GLiNER)  | 68.3%     | 82.2%  | 74.6% |

Takeaways:

- Every OpenMed size beats GLiNER (F1 73–76 vs 65), driven by much higher recall —
  it catches names, cities, states, addresses, and times that regex can't and
  that GLiNER misses more often. Accuracy scales with size (small→base→large).
- `both` matches large's recall using only the small model + GLiNER — useful when
  you can't afford the large model but want coverage.
- Regex alone has the highest precision (79%) but lowest recall (28%): it nails
  structured identifiers and finds nothing free-form. See "Is regex still
  valuable?" below.

Reproduce:

```bash
cargo build --features "ner,bench"
nym bench ai4privacy/pii-masking-300k -n 1000 --ignore-labels --config nym.toml
```

Caveats: label-agnostic span scoring (label-keyed scores need a per-backend label
mapping); ai4privacy is general-domain PII while OpenMed is tuned for
clinical/HIPAA text — a clinical dataset would likely favour it further.

### Is regex still valuable?

Yes — regex and NER are complementary, and regex stays the backbone:

- **Structured, format-defined identifiers** — regex is near-perfect and free:
  email (F1 94.8), IPv4 (98.0), passport (93.7), plus credit cards, IBANs, MAC,
  UUID, JWT, AWS keys, crypto wallets that OpenMed's PII taxonomy doesn't even
  cover. Deterministic, exact boundaries, microseconds, no model, fully offline.
- **The default build has no NER at all** — NER is an opt-in feature; regex is the
  only detector for most users.
- **Free-form entities** (names, cities, states, streets, occupations) are where
  regex scores ~0 and NER is essential.
- **Caveat:** a few regex numeric patterns over-fire on this dataset (e.g. `ssn`,
  `phone_us` generate many false positives on ambiguous digit strings) — context-
  aware NER is more precise there. Tightening those patterns, or letting NER
  arbitrate, would help.

Net: regex carries structured identifiers at ~100% precision; NER carries
free-form entities; merged ("both") gives the best recall. Neither replaces the
other.

### Quantization caveat

Quantizing these DeBERTa-v2 models below fp32 is hard. Verified results:

| Method | Small 44M | Base 184M / Large 434M |
|--------|-----------|------------------------|
| dynamic int8 | ✅ works (566→172 MB, no acc loss) | ❌ collapses (one label per token) |
| dynamic int8, per-channel | ✅ | ❌ collapses |
| static int8 QDQ + calibration | not needed | ❌ collapses |
| fp16 (post-hoc onnxconverter) | — | ❌ broken graph (mixed-precision type errors) |

The disentangled-attention relative-position math is quantization-sensitive, so
int8 destroys base/large regardless of scheme. fp16 would need a **fp16 re-export
from PyTorch** (CUDA) rather than post-hoc conversion — untried.

**Recommendation:** ship **fp32 for base/large** (they're the accurate, recommended
models anyway) and use **int8 only for the small model**. Quantization is therefore
**off by default** (`NYM_QUANTIZE=1` to enable); always validate with `nym detect`.
The base/large benchmarks above use fp32.

### GLiNER vs OpenMed — keep both?

**Yes — keep GLiNER, but OpenMed is the better default for PII.** They do different
jobs:

- For **standard PII** (the fixed taxonomy: names, addresses, IDs, contacts…),
  OpenMed wins outright — higher F1 at every size, and OpenMed-large alone matches
  the recall of OpenMed-small + GLiNER combined. On this axis GLiNER is redundant.
- GLiNER's irreplaceable value is **zero-shot, arbitrary labels at runtime**. It
  detects entity types OpenMed has no label for, with no retraining — e.g. asking
  for `medication`, `medical condition`, `product` extracts *imatinib*, *chronic
  myeloid leukemia*, *Tesla Model 3*. OpenMed structurally cannot do this; its
  head is fixed to 106 PII classes.

So: default to OpenMed for PII detection/anonymization; keep GLiNER for custom or
domain-specific entity extraction. Running `both` is the best PII recall when you
can't deploy the large OpenMed model.

## Performance (latency / CPU / memory)

Measured with `scripts/perf_bench.py` on an 8-core CPU-only Linux box, release
build, median of 3 runs. "Load" is the fixed per-invocation cost (model load +
ONNX Runtime init), isolated by also timing a one-line input; "inference" is the
remainder on a 50 KB document (~7,100 words, ~13 pages).

| Backend              | Load  | 50 KB wall | Throughput  | CPU   | Peak RSS |
|----------------------|-------|------------|-------------|-------|----------|
| regex only           | ~0s   | 0.02s      | instant     | ~100% | **26 MB** |
| openmed-small (int8) | 0.9s  | **8.4s**   | ~950 words/s| 720%  | **552 MB** |
| openmed-base (fp32)  | 2.2s  | 21.7s      | ~365 words/s| 700%  | 1.5 GB |
| gliner               | 3.3s  | 36.6s      | ~210 words/s| 370%  | 2.2 GB |
| both (small+gliner)  | 4.1s  | 67.0s      | ~115 words/s| 390%  | 2.5 GB |
| openmed-large (fp32) | 5.6s  | 72.0s      | ~110 words/s| 680%  | 3.2 GB |

Scaling is ~linear in text length (chunking). Measured on a **200 KB** doc
(~28k words, 4× the 50 KB input):

| Backend | 50 KB | 200 KB |
|---------|-------|--------|
| regex only | 0.02s | 0.02s |
| openmed-small (int8) | 8.4s | 55s |
| openmed-base (fp32) | 21.7s | 101s |
| gliner | 36.6s | 152s |
| both (small+gliner) | 67s | 196s |
| openmed-large (fp32) | 72s | 275s (~4.6 min) |

Peak RSS is unchanged at 200 KB (model-dominated) — e.g. large still ~3.2 GB,
small-int8 ~551 MB.

Takeaways:

- **Regex is free** (15 ms, 26 MB) — why it stays the always-on default.
- **openmed-small-int8 is the efficiency winner**: ~9× faster than large-fp32,
  ~4× faster than GLiNER, ~5× less RAM, best speed/accuracy/memory trade-off
  (F1 73).
- **large-fp32 is the accuracy ceiling (F1 76) but costs 3.2 GB and ~110 w/s** —
  use only when accuracy matters and resources allow.
- **Memory is flat with text size** — the model dominates; chunking caps
  activation memory, so big files don't blow up RSS.
- Every run pays the load cost because nym is a one-shot CLI; a long-running/batch
  mode would amortize it. All numbers are **CPU-only** — GPU/ANE builds change
  them dramatically (see below).

### Reproduce (Linux or macOS)

```bash
# 1. Build (pick the accelerator for your machine)
cargo build --release --features ner            # CPU (any platform)
cargo build --release --features ner-coreml     # macOS: Apple Neural Engine / GPU
cargo build --release --features ner-cuda       # NVIDIA GPU

# 2. Convert the model(s) you want to test
scripts/convert_openmed_onnx.sh                 # small 44M -> models/...-onnx/

# 3. Write a config per backend, e.g. openmed.toml:
#    [ner]
#    enabled = true
#    backend = "openmed"
#    openmed_model = "/abs/path/to/models/<name>-onnx"
#    threshold = 0.5

# 4. Generate a big input and run the harness
python3 - 50000 > big.txt <<'PY'
import sys
p=("Patient {n}: John Smith, DOB 03/15/1985, MRN 447{n}882, email john{n}@ex.com, "
   "phone (415) 555-7012, SSN 123-45-6789, seen in Chicago by Dr. Alice Johnson.\n")
n=0;s=0;t=int(sys.argv[1])
while s<t:
    line=p.format(n=n);sys.stdout.write(line);s+=len(line);n+=1
PY
python3 scripts/perf_bench.py big.txt openmed.toml gliner.toml both.toml
```

`perf_bench.py` is cross-platform: it reads each child's own rusage (handling the
KB-vs-bytes `ru_maxrss` difference between Linux and macOS). Set `NYM_PERF_RUNS`
to change the repeat count and `NYM_BIN` if your binary isn't at
`./target/release/nym`.

**On Apple Silicon:** build with `--features ner-coreml` to route ONNX Runtime
through CoreML (Neural Engine / GPU). Expect markedly lower latency than these
CPU-only numbers, especially for the base/large fp32 models; peak RSS will be
similar since the weights still load into memory.

## Notes & follow-ups

- int8 quantization is opt-in and only reliable for the small model (see caveat
  above). int8 (dynamic/per-channel/static-QDQ) all collapse base/large; the only
  remaining size-reduction avenue is an fp16 *re-export from PyTorch* (needs CUDA)
  — an open follow-up.
- BIO spans for fragmented numerics (e.g. dates) can split where the model
  alternates labels across sub-tokens; OpenMed's Python "smart merging" is not
  yet ported.
- Models are loaded from a **local directory**. Pulling a pre-converted ONNX repo
  straight from the HF Hub (like the GLiNER path) is a natural next step.
