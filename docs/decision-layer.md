# Decision-model adjudication layer

nym's regex + NER layers are deterministic and fast, but they share the same blind
spot as any regex scanner: **unlabeled secrets** (data that carries no name and no
recognisable structure) and context-dependent "is this actually private?" calls.
`nym decide` adds a System-One style **decision model** that sits *after* the
deterministic layers as the residual gate.

This makes nym usable as a **session-trace scrubber**: detect secrets + PII, then
adjudicate each candidate so you redact the real secrets and veto over-redactions
(code samples, paths, uuids, checksums, example tokens) before the trace is shared
or uploaded (e.g. to Hugging Face). The **originals are never rewritten** — the
deterministic layers report spans, the decision layer annotates them, and a
downstream scrubber decides.

## What it does

- Runs the deterministic detector (regex + optional NER) and a **high-entropy
  backstop** for unlabeled base64/hex-like blobs.
- Asks an OpenAI-compatible `/v1/chat/completions` endpoint a typed question per
  candidate: is this a real secret? The model answers `redact` / `keep` / `flag`
  with a class and a confidence.
- Returns one `Decision` per candidate carrying `verdict`, `class`, `confidence`,
  `source` (`detector` / `entropy`), and the detector's pattern/confidence.

## Build

Requires the `decision` feature (pulls in `ureq`):

```bash
cargo build --release --features "decision,ner"
```

## Run

```bash
# Adjudicate detected spans in a file
nym decide trace.txt \
  --endpoint http://100.64.0.26:8001/v1/chat/completions \
  --model deepseek-v4-flash-vision \
  --threshold 0.5 --output-json
```

Text output is a human-readable table; `--output-json` is machine-readable
(`[{verdict, class, confidence, source, text, start, end, ...}]`).

### Flags

| Flag | Meaning |
|------|---------|
| `--endpoint` | OpenAI-compatible chat-completions URL (overrides `[decision] endpoint`) |
| `--model`    | Model id (overrides `[decision] model`) |
| `--threshold`| p(secret) at/above which a candidate is adjudicated `redact` (default 0.5) |
| `--max-candidates` | Cap on candidates adjudicated in one run (0 = unlimited) |
| `--output-json` | Machine-readable JSON output |

### Configuration

```toml
[decision]
enabled = true
endpoint = "http://100.64.0.26:8001/v1/chat/completions"
model = "deepseek-v4-flash-vision"
api_key_env = "NYM_DECISION_KEY"   # optional, read at runtime, never stored
threshold = 0.5
context_chars = 160                 # context window sent per candidate
batch_size = 1                      # one candidate per request (most reliable)
entropy_backstop = true             # add unlabeled high-entropy candidates
```

### Choosing a backend

| Backend | Notes |
|---------|-------|
| `chat` (default) | OpenAI-compatible `/v1/chat/completions`; label-only. The model's `confidence` is a decoded token, so it is NOT calibrated. Best for shipping a gate, not for a trust threshold. |
| `systemone` | TypeSafe/Jev-compatible `/v1/systemone` readout (Choice/Noul/Score -> true per-option `probabilities` + derived, calibrated `confidence`). Use Kev-4B/Decider on your own GPU, or the official TypeSafe API. This is the calibrated path. |

A capable chat model (deepseek-flash on rtx6000, etc.) is a fine `chat` backend
(~1s/candidate). A small local model (Qwen2.5-1.5B on a 4090 via llama.cpp) is
~0.8s/candidate but notably lower accuracy on ambiguous classes.

Accuracy matters more than raw speed for a decision gate: a 1.5B model confuses a
context length (`262144`) or a place name (`Chicago`) for a secret, while a
stronger model corrects them. Use the calibrated `systemone` backend (Kev) when
you need to threshold/stage a failing-closed gate; the `confidence` is derived
from the probability distribution, not a decoded token.

#### systemone (Kev / TypeSafe)

```bash
# Kev-4B on the 4090 (kev.serve):
nym decide input.txt \
  --backend systemone \
  --endpoint http://100.64.0.9:8009 \
  --model kev-latest

# Official TypeSafe/Jev API (same contract, different base URL + key):
# set NYM_DECISION_KEY (or whatever api_key_env is in [decision]) before running
TYPESAFE_API_KEY=... nym decide input.txt \
  --backend systemone \
  --endpoint https://api.typesafe.ai \
  --model jev-latest
```

The endpoint may be a bare base URL (`http://host:8009`) or already include the
path; `/v1/systemone` is appended when needed. The gate sends the chunk as
`state` with one Choice per candidate (`redact`/`keep`/`flag`), and reads the
calibrated answer back. No external crate is required.

## Design notes

- `batch_size = 1` is the default because many served models collapse a
  multi-candidate request into one JSON object. Single-candidate requests are
  answered reliably and can be parsed defensively.
- The API key is read from an environment variable named by `api_key_env`; it is
  never stored in the repo or the response.
- Only the bounded `context_chars` window around each candidate leaves the
  machine — never the whole document (privacy/no-egress).
- The model is a **gate/adjudicator**, not a security boundary. It is a
  high-recall gauntlet against *honest* misses; pair it with the deterministic
  layers for the structural guarantees.

## Labeling loop

To build a real labeled set for tuning the decision layer, chop chat-session
traces into at-a-glance chunks, pre-label them with nym, and correct the
pre-labels in a swipe UI:

```bash
python3 scripts/label_sessions.py --source pi --source hstry \
  --bucket nym-sensitive --chunk-chars 300 --shuffle --sensitive-first \
  --endpoint http://100.64.0.26:8001/v1/chat/completions \
  --model deepseek-v4-flash-vision
```

Then open `http://100.64.0.12:8511/sens`. `--source pi` reads
`~/.pi/agent/sessions` directly; `--source hstry` exports conversations from the
[hstry](https://github.com/byteowlz/hstry) history database (reaching opencode,
codex, claude-code, chatgpt, ... even when the original files are not pi-format).
`--sensitive-first` surfaces nym-predicted-sensitive chunks first so the
important corrections precede the long benign tail. Every card carries
`meta.nym_prediction`, so your corrections are a direct accuracy/recall signal
for the decision layer; pull them back with `GET /api/sens/results/<bucket>`.