#!/usr/bin/env python3
"""Chop chat-session traces into judge-at-a-glance chunks, pre-label them with
nym (deterministic detector + decision gate), and feed them as datatinder swipe
candidates (good = "contains sensitive data", bad = "does not").

The point is a high-ergonomics labeling loop for YOUR real sessions:
- we pre-label each chunk, so you correct the pre-label instead of judging
  from scratch;
- each card is short enough to judge at a glance;
- a `meta.nym_prediction` records what nym thought, so your corrections are a
  direct accuracy/recall signal for the nym decision layer.

Two sources are supported and can be mixed (`--source` is repeatable):
- `pi`     : the pi session files under `--sessions-dir`
             (default ~/.pi/agent/sessions). These are read directly.
- `hstry`  : the hstry canonical history database (`hstry export --format pi`).
             This reaches every agent hstry has indexed (pi, opencode, codex,
             claude-code, chatgpt, ...) even when the original files are not a
             pi-format JSONL. `--source hstry` with `--agents-source <name>`
             narrows to one indexed source.

Usage:
    python3 scripts/tinder_sessions.py --source pi --source hstry \
        --bucket nym-sensitive --chunk-chars 300 --max-sessions 50 --shuffle \
        --dry-run
    # add pre-labeling from a decision endpoint:
    ... --endpoint http://100.64.0.26:8001/v1/chat/completions \
        --model deepseek-v4-flash-vision

The bucket root is the datatinder data root (default
~/byteowlz/datatinder/data). It writes <bucket>/swipe.jsonl which the datatinder
swipe UI autoloads. Cards carry meta {nym_prediction, source, session,
chunk_index}, so a later pull of /api/swipe/results/<bucket> has the pre-label
attached to every correction.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_DATA_ROOT = Path(os.path.expanduser("~/byteowlz/datatinder/data"))
DEFAULT_SESSIONS = Path(os.path.expanduser("~/.pi/agent/sessions"))
NYM_BIN = str(Path(__file__).resolve().parent.parent / "target" / "release" / "nym")


# --------------------------------------------------------------------------
# pi-format session reading
# --------------------------------------------------------------------------

def read_jsonl(path):
    entries = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return entries


def iter_pi_sessions(sessions_dir: Path):
    """Yield (jsonl_path, entries) for every *.jsonl under sessions_dir."""
    for p in sorted(sessions_dir.rglob("*.jsonl")):
        entries = read_jsonl(p)
        if entries:
            yield p, entries


def extract_text_blocks(entries):
    """Yield (role, text) for each user/assistant/tool content block.

    pi session entries have `type` in {message,...}; a `message` entry has
    `message.content` = a list of blocks of type `text`, `thinking`,
    `toolCall`, or `image`. The `message.role` is `user|assistant|toolResult`.

    We surface user + assistant text, tool-call arguments, and tool results
    (the richest sensitive surface: bash output, file contents, URLs) but skip
    pure `thinking` blocks (often long/opaque and rarely the secret surface)
    and `image` blocks.
    """
    for e in entries:
        if e.get("type") != "message":
            continue
        msg = e.get("message") or {}
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, str):
            yield role or "toolResult", content
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype in ("text", "output_text") and block.get("text"):
                yield role or "message", block["text"]
            elif btype == "thinking" and block.get("thinking"):
                # thinking is often long and rarely the sensitive surface;
                # skip unless short enough to judge (keeps cards ergonomic).
                pass
            elif btype in ("toolCall", "tool_use") and block.get("name"):
                args = block.get("arguments") or block.get("input")
                if isinstance(args, dict):
                    args = json.dumps(args)
                yield role or "tool", f"tool:{block['name']} {args}"
            elif btype == "tool_result" and block.get("content"):
                c = block["content"]
                if isinstance(c, list):
                    c = " ".join(
                        b.get("text", "") for b in c if isinstance(b, dict)
                    )
                yield role or "toolResult", str(c)


def split_chunks(text: str, chunk_chars: int):
    """Split a text block into at-a-glance chunks at whitespace boundaries."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    while text:
        if len(text) <= chunk_chars:
            chunks.append(text)
            break
        cut = text.rfind(" ", 0, chunk_chars)
        if cut <= 0:
            cut = chunk_chars
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    return [c for c in chunks if c]


# --------------------------------------------------------------------------
# hstry source
# --------------------------------------------------------------------------

def hstry_conversations(agents_source=None, limit=None):
    """Return conversation metadata rows from `hstry list --json`."""
    cmd = ["hstry", "list", "--json"]
    if agents_source:
        cmd += ["--source", agents_source]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    try:
        payload = json.loads(out.stdout or "{}")
        rows = payload.get("result") or {}
    except Exception:
        return []
    return list(rows)[:limit] if isinstance(rows, list) else []


def hstry_export_to_pi(convs, out_dir: Path):
    """Export hstry conversations to pi-format JSONL in out_dir.

    Returns a list of (jsonl_path, source_id, title). `hstry export --format pi`
    writes one session file per conversation under <out_dir>/sessions/.
    """
    ids = [c["id"] for c in convs if c.get("id")]
    if not ids:
        return []
    cmd = ["hstry", "export", "--format", "pi", "-c", ",".join(ids), "-o", str(out_dir)]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except Exception:
        return []
    meta = {c["id"]: c for c in convs}
    files = []
    sessions_dir = out_dir / "sessions"
    if sessions_dir.is_dir():
        for p in sessions_dir.rglob("*.jsonl"):
            # Match by the conversation id embedded in the exported filename,
            # else fall back to the title/source.
            files.append((p, meta))
    return files


# --------------------------------------------------------------------------
# pre-labeling with nym
# --------------------------------------------------------------------------

def _apply_pred(row, decisions):
    """Compute a pre-label dict from nym's decisions and store it on a row."""
    verdicts = [d.get("verdict") for d in decisions]
    any_redact = "redact" in verdicts
    # For stratification and judge pre-fill, capture the class / PII
    # sub-types nym assigned to ANY span (including ones it judged keep),
    # not just redacted ones -- the keep-but-really-sensitive surface is
    # exactly what we want to surface to the labeler.
    nym_class = None
    for d in decisions:
        c = d.get("class") or ""
        if c:
            nym_class = c
            break
    pii_subs = []
    for d in decisions:
        sub = pii_sub_class(d.get("class"))
        if sub and sub not in pii_subs:
            pii_subs.append(sub)
    row["_pred"] = {
        "nym_prediction": "sensitive" if any_redact else "not_sensitive",
        "nym_verdicts": verdicts,
        "n_spans": len(decisions),
        "class": nym_class,
        "pii_subs": pii_subs,
    }


def prelabel(rows, nym_bin, endpoint, model, threshold, backend=None, fast=False):
    """Pre-label each chunk with nym `decide`.

    No-op if endpoint is unset (the loop still works unlabeled).

    Uses `--jsonl` batch mode so all chunks go through a single `nym` process,
    amortizing the startup + detector init (one process per chunk is ~0.9s of
    pure overhead). Chunks are processed line-by-line; results map back to rows.
    `backend` is passed through to `nym decide` (default: the `chat` backend).
    `fast` writes a temp config that sets a large `batch_size` (so all of a
    chunk's candidates go in one request) and disables the high-entropy
    backstop (the expensive per-chunk scan) -- for the labeling sweep.
    """
    if not endpoint:
        return
    config_path = None
    if fast:
        cfg = (
            "[decision]\n"
            "enabled = true\n"
            "batch_size = 4096\n"  # all candidates of a chunk in one request
            "entropy_backstop = false\n"  # skip the expensive per-chunk scan
        )
        config_path = "/tmp/_nym_fast.toml"
        open(config_path, "w").write(cfg)
    # Write chunks as line-delimited JSON, in row order, to a temp file.
    batch = "\n".join(json.dumps({"text": r["_text"]}) for r in rows)
    infile = "/tmp/_nym_batch.jsonl"
    open(infile, "w").write(batch)
    cmd = [nym_bin, "decide", "--jsonl",
           "--endpoint", endpoint, "--model", model,
           "--threshold", str(threshold)]
    if backend:
        cmd += ["--backend", backend]
    if config_path:
        cmd += ["--config", config_path]
    # Output is one JSON object per input line, in order.
    try:
        out = subprocess.run(cmd, stdin=open(infile), capture_output=True, text=True, timeout=7200)
        lines = [l for l in out.stdout.splitlines() if l.strip()]
    except Exception:
        lines = []
    for i, row in enumerate(rows):
        if i >= len(lines):
            continue
        try:
            decisions = json.loads(lines[i])
        except Exception:
            continue
        _apply_pred(row, decisions)


def pii_sub_class(cls: str):
    """Map nym's free-text class onto a canonical PII sub-tag (or None if it
    is not PII).
    """
    if not cls:
        return None
    l = cls.lower()
    if "email" in l:
        return "email"
    if "phone" in l or "mobile" in l or "tel" in l:
        return "phone"
    if "ssn" in l or "national" in l or "passport" in l or "driver" in l or "license" in l:
        return "id"
    if "address" in l or "street" in l or "city" in l or "geo" in l or "zip" in l:
        return "address"
    if "dob" in l or "birth" in l or "date" in l or "age" in l:
        return "dob"
    if "username" in l or "handle" in l or "user" in l:
        return "username"
    if "company" in l or "org" in l:
        return "org"
    if "pii" in l or "name" in l or "person" in l:
        return "name"
    return "other"


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def gather_pi_rows(sessions_dir, rng, args):
    rows = []
    paths = list(iter_pi_sessions(sessions_dir))
    if args.shuffle:
        rng.shuffle(paths)
    paths = paths[: args.max_sessions]
    for path, entries in paths:
        chunks = []
        for _role, text in extract_text_blocks(entries):
            chunks.extend(split_chunks(text, args.chunk_chars))
        if args.chunks_per_session > 0:
            rng.shuffle(chunks)
            chunks = chunks[: args.chunks_per_session]
        for i, ch in enumerate(chunks):
            rows.append({"_text": ch, "id": f"{path.stem}_{i}",
                         "session": path.stem, "chunk_index": i,
                         "source": "pi"})
    return rows


def gather_hstry_rows(rng, args):
    rows = []
    convs = hstry_conversations(args.agents_source, args.max_sessions)
    if not convs:
        return rows
    rng.shuffle(convs)
    with tempfile.TemporaryDirectory(prefix="nym-hstry-") as td:
        exported = hstry_export_to_pi(convs, Path(td))
        for path, _meta in exported:
            entries = read_jsonl(path)
            chunks = []
            for _role, text in extract_text_blocks(entries):
                chunks.extend(split_chunks(text, args.chunk_chars))
            if args.chunks_per_session > 0:
                rng.shuffle(chunks)
                chunks = chunks[: args.chunks_per_session]
            for i, ch in enumerate(chunks):
                rows.append({"_text": ch, "id": f"{path.stem}_{i}",
                             "session": path.stem, "chunk_index": i,
                             "source": "hstry"})
    return rows


def heuristic_interest(text):
    """Independent structural interest score for stratification.

    nym's decision gate under-detects on dev-trace data (it found zero spans
    across many sessions), so we cannot rely on n_spans to surface the
    recall-error surface. This regex scorer runs off the raw text and catches
    the sensitive-in-dev-trace shapes a human would flag: internal paths,
    usernames/emails, IPs/hostnames, and key/token/BEARER-like strings.

    Returns (score, matched_names) where score is an int weight and
    matched_names is the list of signals that fired, for display in the judge.
    """
    import re
    matched = []
    score = 0
    pats = {
        "path":     r"(?:/home/|/Users/|/var/|/etc/|/opt/|\\Users\\|C:\\|\\n|src/|/workspace)",
        "user":     r"\b(?:user(?:name)?|admin|root|operator|svc_?[a-z]|_svc)\b",
        "email":    r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}",
        "ip":       r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "host":     r"\b(?:[a-z0-9-]+\.)+(?:com|net|org|io|dev|local|internal|corp|lan)",
        "token":    r"(?:api[_-]?key|secret|token|bearer|authoriz|passw|pwd|aws_|AKIA|BEGIN [A-Z ]+KEY)",
        "uuid":     r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        "hex":      r"\b[0-9a-f]{16,}\b",
        "b64":      r"\b[A-Za-z0-9+/]{24,}={0,2}\b",
        "jwt":      r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b",
        "date":     r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
    }
    for name, pat in pats.items():
        if re.search(pat, text):
            matched.append(name)
            score += {"path": 3, "user": 3, "email": 4, "ip": 3, "host": 2,
                      "token": 5, "uuid": 2, "hex": 2, "b64": 2, "jwt": 5,
                      "date": 1}.get(name, 1)
    return min(score, 12), matched


def balanced_select(cards, rng, args):
    """Return a stratified, high-signal subset for human labeling.

    The naive random corpus is ~99% obvious negatives, so most labels confirm
    'not sensitive' and tell us nothing about the scrubber's recall. Instead
    we concentrate effort on the cases that actually change nym's behavior.

    Because nym's decision gate under-detects on dev traces (n_spans is often
    0 even on sensitive-looking text), we stratify primarily on an independent
    structural heuristic (heuristic_interest) over the raw text:

      Tier A: nym-predicted sensitive (true positives to confirm).
      Tier B: nym found a span but judged it keep.
      Tier C: chunks the heuristic flags as interesting (paths, usernames,
              emails, IPs, tokens) that nym may have missed -- the recall-error
              surface -- plus a small stratified slice of the benign tail so
              precision stays calibrated.

    --require-class filters chunks by the nym class if given. --max-cards caps
    the total (benign slice shrinks to fit).
    """
    def interest(c):
        return c["meta"].get("_interest", 0)

    tier_a = [c for c in cards if c["meta"].get("nym_prediction") == "sensitive"]
    tier_b = [c for c in cards
              if c["meta"].get("nym_prediction") != "sensitive"
              and c["meta"].get("n_spans")]
    # Everything else, ranked by heuristic interest, split into an
    # interesting upper band (recall surface) and a benign lower band.
    rest = [c for c in cards
            if c not in tier_a and c not in tier_b]
    for c in rest:
        sc, why = heuristic_interest(c["text"])
        c["meta"]["_interest"] = sc
        c["meta"]["_why"] = why
    rest.sort(key=lambda c: (-interest(c), c["text"]))
    interesting = [c for c in rest if interest(c) >= 4]
    benign_pool = [c for c in rest if interest(c) < 4]

    # Build the set: all of A + B + the interesting band, then take a
    # benign fraction of the benign pool for calibration.
    chosen = tier_a + tier_b + interesting
    rng.shuffle(benign_pool)
    benign_keep = 0
    if args.max_cards is not None:
        benign_keep = max(0, args.max_cards - len(chosen))
    else:
        benign_keep = int(round(len(benign_pool) * args.benign_fraction))
    chosen += benign_pool[:benign_keep]

    # Honor --require-class by filtering the final set (but keep at least a
    # representative benign slice so the classifier still sees negatives).
    if args.require_class:
        def hit(c):
            cls = (c["meta"].get("class") or "").lower()
            return any(r in cls for r in args.require_class)
        kept = [c for c in chosen if hit(c)]
        # Always retain a small clean subset for calibration.
        kept += [c for c in chosen if not hit(c)][:max(0, benign_keep)]
        chosen = kept
    return chosen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", action="append", default=None,
                    help="source(s) to pull from: 'pi' and/or 'hstry' (repeatable; default: pi)")
    ap.add_argument("--agents-source", default=None,
                    help="narrow hstry to one indexed source, e.g. opencode/codex/pi")
    ap.add_argument("--sessions-dir", default=str(DEFAULT_SESSIONS))
    ap.add_argument("--bucket", default="nym-sensitive")
    ap.add_argument("--out", default=None)
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--chunk-chars", type=int, default=300)
    ap.add_argument("--max-sessions", type=int, default=50,
                    help="max sessions/conversations per source")
    ap.add_argument("--chunks-per-session", type=int, default=20)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sensitive-first", action="store_true",
                    help="order cards so nym-predicted-sensitive (or high-entropy) chunks come first; the important corrections surface before the long benign tail")
    ap.add_argument("--balanced", action="store_true",
                    help="stratified high-signal sample: keep every nym-predicted-sensitive card, "
                         "every card where nym found a span but judged it keep (the recall-error "
                         "surface), plus a stratified slice of the benign tail. Targets the cases "
                         "that actually change the scrubber instead of the obvious negatives.")
    ap.add_argument("--max-cards", type=int, default=None,
                    help="cap total balanced cards (default: unlimited; benign tail slice shrinks to fit)")
    ap.add_argument("--benign-fraction", type=float, default=0.15,
                    help="balanced mode: fraction of the benign (zero-span) tail to include (default 0.15)")
    ap.add_argument("--require-class", action="append", default=None,
                    help="balanced mode: only keep chunks whose nym class contains one of these strings (repeatable), e.g. --require-class username --require-class path")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--endpoint", default=None, help="decision endpoint for pre-labeling")
    ap.add_argument("--model", default=None, help="decision model id")
    ap.add_argument("--backend", default=None, help="decision backend for pre-labeling (chat | systemone)")
    ap.add_argument("--entropy-backstop", action="store_true",
                    help="enable the high-entropy unlabeled-secret backstop during pre-labeling (default off for speed)")
    ap.add_argument("--config", default=None, help="path to a nym config toml (overrides for pre-labeling)")
    ap.add_argument("--fast", action="store_true",
                    help="pre-label fast: batch all candidates per chunk into one request (large batch_size) and turn the high-entropy backstop off. Use for the labeling sweep; keep entropy on for the real scrub pass.")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--nym", default=NYM_BIN)
    args = ap.parse_args(argv)

    root = Path(os.path.expanduser(args.data_root))
    out = Path(args.out) if args.out else root / args.bucket / "swipe.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    sources = args.source or ["pi"]
    rows = []
    if "pi" in sources:
        sessions_dir = Path(os.path.expanduser(args.sessions_dir))
        if sessions_dir.is_dir():
            rows += gather_pi_rows(sessions_dir, rng, args)
        else:
            print(f"pi sessions dir not found: {sessions_dir}", file=sys.stderr)
    if "hstry" in sources:
        rows += gather_hstry_rows(rng, args)

    if not rows:
        print("no chunks produced from the given sources", file=sys.stderr)
        return 1

    prelabel(rows, args.nym, args.endpoint, args.model, args.threshold, args.backend, fast=args.fast)

    # De-duplicate by text so the same span isn't judged twice.
    seen = set()
    cards = []
    for r in rows:
        if r["_text"] in seen:
            continue
        seen.add(r["_text"])
        pred = r.pop("_pred", {})
        meta = {"session": r["session"], "chunk_index": r["chunk_index"],
                "chunk_chars": len(r["_text"]), "source": r["source"], **pred}
        cards.append({"id": r["id"], "context": f"[{r['source']} {r['session']}]",
                      "text": r["_text"], "meta": meta})

    if args.sensitive_first:
        # Sensitive-first: nym-predicted-sensitive chunks come first, then
        # any chunk with a detection (n_spans>0), then the benign tail. Within
        # each bucket keep the existing order (shuffle or file order).
        def bucket_key(c):
            m = c["meta"]
            if m.get("nym_prediction") == "sensitive":
                return 0
            if m.get("n_spans"):
                return 1
            return 2
        cards.sort(key=bucket_key)

    if args.balanced:
        cards = balanced_select(cards, rng, args)

    if args.dry_run:
        n_pred = sum(1 for c in cards if c["meta"].get("nym_prediction"))
        print(f"would write {len(cards)} cards to {out} "
              f"({n_pred} pre-labeled, {len(cards)-n_pred} unlabeled)")
        return 0

    existing_ids = set()
    if out.exists():
        for line in open(out):
            line = line.strip()
            if line:
                try:
                    existing_ids.add(json.loads(line).get("id"))
                except Exception:
                    continue
    written = 0
    with open(out, "a") as fh:
        for c in cards:
            if c["id"] in existing_ids:
                continue
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
            written += 1
    n_pred = sum(1 for c in cards if c["meta"].get("nym_prediction"))
    print(f"wrote {written} new cards to {out} (total {len(cards)}, "
          f"{n_pred} pre-labeled). Swipe at http://100.64.0.12:8511/sens "
          f"bucket={args.bucket}")
    return 0


if __name__ == "__main__":
    sys.exit(main())