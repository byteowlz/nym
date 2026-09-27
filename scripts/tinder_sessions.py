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

def prelabel(rows, nym_bin, endpoint, model, threshold):
    """Pre-label each chunk with nym `decide`.

    No-op if endpoint is unset (the loop still works unlabeled).
    """
    if not endpoint:
        return
    for row in rows:
        txt = row["_text"]
        open("/tmp/_nym_row.txt", "w").write(txt)
        cmd = [nym_bin, "decide", "/tmp/_nym_row.txt",
               "--endpoint", endpoint, "--model", model,
               "--threshold", str(threshold), "--output-json"]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            decisions = json.loads(out.stdout or "[]")
        except Exception:
            continue
        verdicts = [d.get("verdict") for d in decisions]
        any_redact = "redact" in verdicts
        # The class nym assigned to a redacted span, for pre-filling the
        # judge's class picker (e.g. "credential/secret", "pii", ...).
        redacted = [d for d in decisions if d.get("verdict") == "redact"]
        nym_class = None
        for d in redacted:
            c = d.get("class") or ""
            if c:
                nym_class = c
                break
        row["_pred"] = {
            "nym_prediction": "sensitive" if any_redact else "not_sensitive",
            "nym_verdicts": verdicts,
            "n_spans": len(decisions),
            "class": nym_class,
        }


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
                    help="order cards so nym-predicted-sensitive (or high-entropy) chunks come first; \
                         the important corrections surface before the long benign tail")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--endpoint", default=None, help="decision endpoint for pre-labeling")
    ap.add_argument("--model", default=None, help="decision model id")
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

    prelabel(rows, args.nym, args.endpoint, args.model, args.threshold)

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
          f"{n_pred} pre-labeled). Swipe at http://100.64.0.12:8511/swipe "
          f"bucket={args.bucket}")
    return 0


if __name__ == "__main__":
    sys.exit(main())