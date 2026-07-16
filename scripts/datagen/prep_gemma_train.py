#!/usr/bin/env python3
"""Prepare the gemma-labeled real-text corpus for training: optionally
down-sample an over-represented label, then split train/val.

Why down-sample DATE: Wikipedia is date-saturated, so the LLM-labeled corpus
comes out ~36% DATE spans -- more than GIVEN_NAME. That skews the loss toward a
label of low anonymization value, and over-flagged dates cost precision on
benchmark gold that only marks dates-of-birth.

This is safe *only* because these rows train with `--mask-o-sources wikipedia`:
a dropped DATE span becomes an O token, and weak-source O tokens are masked out
of the loss (-100). So dropping a span removes signal rather than creating a
false negative.

The one trap: a weak row whose spans are ALL dropped becomes entity-free, which
makes it eligible for `--weak-neg-keep` promotion to full O supervision -- i.e.
it would actively teach "dates are not PII" and fight the synthetic data. Such
rows contribute zero signal anyway (weak row, no positive spans), so they are
dropped outright. Rows that were *already* PII-free are left alone: they are the
intended negative pool.
"""
import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=Path, default=Path("data/gemma/gemma.jsonl"))
    ap.add_argument("--out-prefix", default="data/gemma2")
    ap.add_argument("--label", default="DATE", help="label to down-sample")
    ap.add_argument("--keep", type=float, default=0.25,
                    help="fraction of that label's spans to keep (1.0 = no change)")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    rows = [r for r in rows if not r.get("__done__")]

    kept_rows, dropped_noop, before, after = [], 0, 0, 0
    for r in rows:
        ents = r.get("entities") or []
        before += sum(1 for e in ents if e["label"] == args.label)
        if ents and args.keep < 1.0:
            new = [e for e in ents
                   if e["label"] != args.label or rng.random() < args.keep]
            if not new:
                # every span was a down-sampled DATE: as a weak row this now
                # supervises nothing, and if promoted it would teach dates=O.
                dropped_noop += 1
                continue
            r["entities"] = new
        after += sum(1 for e in (r.get("entities") or []) if e["label"] == args.label)
        kept_rows.append(r)

    rng.shuffle(kept_rows)
    n_val = int(len(kept_rows) * args.val_frac)
    val, train = kept_rows[:n_val], kept_rows[n_val:]

    for name, part in (("train", train), ("val", val)):
        p = Path(f"{args.out_prefix}.{name}.jsonl")
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {p}: {len(part)} rows")

    total = sum(len(r.get("entities") or []) for r in kept_rows)
    print(f"{args.label} spans: {before} -> {after} "
          f"({100 * after / max(total, 1):.1f}% of {total} spans)")
    print(f"rows: {len(rows)} -> {len(kept_rows)} "
          f"(dropped {dropped_noop} that lost every span)")


if __name__ == "__main__":
    main()
