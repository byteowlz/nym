#!/usr/bin/env python3
"""Fine-tune a multilingual token-classification PII model for nym.

Consumes the char-offset JSONL produced by scripts/datagen (no separate BIO step
needed — labels are aligned to the model's own tokenizer here, so there's never a
tokenizer mismatch). Trains an AutoModelForTokenClassification (default:
jhu-clsp/mmBERT-base — a modern ModernBERT-architecture multilingual encoder) and
saves it ready for ONNX export via scripts/convert_openmed_onnx.sh.

GPU strongly recommended (run this on the box with the GPU, e.g. hp-z8).

Example:
  uv run --with "transformers>=4.48" --with torch --with datasets --with seqeval \
    --with accelerate scripts/train_ner.py \
    --train "data/pii*.train.jsonl" --val "data/pii*.val.jsonl" \
    --base-model jhu-clsp/mmBERT-base --output-dir models/nym-pii-mmbert \
    --epochs 3 --batch-size 16 --max-length 256

Dry run (no GPU, verifies data + label alignment):
  uv run --with "transformers>=4.48" --with torch scripts/train_ner.py \
    --train "data/pii*.train.jsonl" --dry-run
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


def load_jsonl(patterns):
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    if not files:
        sys.exit(f"no files matched: {patterns}")
    rows = []
    for f in files:
        with open(f) as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    sys.stderr.write(f"loaded {len(rows)} examples from {len(files)} file(s)\n")
    return rows


def build_labels(rows):
    types = sorted({e["label"] for r in rows for e in r["entities"]})
    labels = ["O"] + [f"{p}-{t}" for t in types for p in ("B", "I")]
    return labels, {l: i for i, l in enumerate(labels)}


def align(rows, tokenizer, label2id, max_length, mask_o_sources=frozenset()):
    """Tokenize each example and assign a label id per token from char spans.
    First token of an entity -> B-, subsequent overlapping tokens -> I-, tokens
    outside any entity -> O, special/pad tokens -> -100 (ignored in the loss).

    mask_o_sources: `source` values with WEAK labels (e.g. teacher-labeled real
    text, where an undetected entity would otherwise train as a false "O").
    For those records, O tokens become -100 — only positive spans supervise."""
    def gen():
        for r in rows:
            weak = r.get("source") in mask_o_sources
            enc = tokenizer(r["text"], truncation=True, max_length=max_length,
                            return_offsets_mapping=True)
            labels = []
            ents = sorted(r["entities"], key=lambda e: e["start"])
            for (a, b) in enc["offset_mapping"]:
                if a == b:  # special token
                    labels.append(-100)
                    continue
                tag = "O"
                for e in ents:
                    if a < e["end"] and b > e["start"]:  # overlap
                        tag = ("B-" if a <= e["start"] else "I-") + e["label"]
                        break
                if tag == "O" and weak:
                    labels.append(-100)
                else:
                    labels.append(label2id.get(tag, label2id["O"]))
            enc.pop("offset_mapping")
            enc["labels"] = labels
            yield enc
    return list(gen())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True, help="JSONL path(s)/glob(s)")
    ap.add_argument("--val", nargs="+", default=None)
    ap.add_argument("--test", nargs="+", default=None)
    ap.add_argument("--base-model", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--output-dir", default="models/nym-pii-mmbert")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--optim", default="adamw_torch", help="e.g. adamw_torch, adamw_bnb_8bit (low-VRAM)")
    ap.add_argument("--grad-checkpointing", action="store_true", help="trade compute for memory")
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--mask-o-sources", default=None,
                    help="comma list of `source` values whose O tokens are ignored in the "
                         "loss (weak/teacher labels: only positive spans supervise); "
                         "their PII-free records are dropped entirely")
    ap.add_argument("--weak-neg-keep", type=float, default=0.0,
                    help="fraction of PII-free weak-source records to keep WITH normal O "
                         "supervision (restores real-prose precision; the poison risk is "
                         "confined to this fraction). 0 = drop all (recall-max).")
    ap.add_argument("--o-weight", type=float, default=1.0,
                    help="CE weight of the O class. <1 makes a missed entity cost more "
                         "than a false flag (recall-tilted training).")
    ap.add_argument("--distill-from", default=None,
                    help="teacher checkpoint for online logit distillation. Must share "
                         "the student's tokenizer and label map. KD covers ALL attended "
                         "tokens, so the teacher also supervises weak-masked O positions.")
    ap.add_argument("--distill-alpha", type=float, default=0.5,
                    help="loss = (1-a)*CE + a*KL(student||teacher)")
    ap.add_argument("--distill-temp", type=float, default=2.0)
    ap.add_argument("--dry-run", action="store_true", help="load+align a sample, print, exit (no training)")
    args = ap.parse_args()
    mask_srcs = frozenset(s.strip() for s in args.mask_o_sources.split(",")) if args.mask_o_sources else frozenset()

    from transformers import AutoTokenizer

    train_rows = load_jsonl(args.train)
    if mask_srcs:
        import random as _random
        rng = _random.Random(11)
        n0, kept_neg = len(train_rows), 0
        rows = []
        for r in train_rows:
            if r.get("source") in mask_srcs and not r["entities"]:
                if rng.random() < args.weak_neg_keep:
                    r = dict(r)
                    r.pop("source")  # promote: full (unmasked) O supervision
                    kept_neg += 1
                    rows.append(r)
            else:
                rows.append(r)
        train_rows = rows
        sys.stderr.write(f"mask-o-sources {sorted(mask_srcs)}: kept {kept_neg} PII-free weak "
                         f"records with O supervision (weak-neg-keep={args.weak_neg_keep}), "
                         f"dropped {n0 - len(train_rows)}; O masked on weak positives\n")
    labels, label2id = build_labels(train_rows)
    id2label = {i: l for l, i in label2id.items()}
    sys.stderr.write(f"{len(labels)} BIO labels over {(len(labels)-1)//2} entity types\n")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    if args.dry_run:
        sample = align(train_rows[:200], tokenizer, label2id, args.max_length)
        # Show one aligned example.
        ex = next(s for s in sample if any(l not in (-100, label2id["O"]) for l in s["labels"]))
        toks = tokenizer.convert_ids_to_tokens(ex["input_ids"])
        sys.stderr.write("\nsample alignment (non-O tokens):\n")
        for t, l in zip(toks, ex["labels"]):
            if l not in (-100, label2id["O"]):
                sys.stderr.write(f"  {t:20s} {id2label[l]}\n")
        # sanity: coverage of -100 vs labeled
        import statistics
        lens = [len(s["input_ids"]) for s in sample]
        sys.stderr.write(f"\nok: {len(sample)} aligned, median tokens {int(statistics.median(lens))}\n")
        sys.stderr.write(f"labels: {labels[:6]} ... ({len(labels)} total)\n")
        return

    import numpy as np
    from transformers import (AutoModelForTokenClassification, DataCollatorForTokenClassification,
                              Trainer, TrainingArguments)

    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model, num_labels=len(labels), id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True)  # layer-dropped inits may carry a differently-sized head

    teacher = None
    if args.distill_from:
        teacher = AutoModelForTokenClassification.from_pretrained(args.distill_from).eval()
        if teacher.config.label2id != label2id:
            sys.exit(f"teacher label map differs from student's ({args.distill_from}); "
                     f"KD logits would supervise the wrong classes")
        for p in teacher.parameters():
            p.requires_grad_(False)

    train_ds = align(train_rows, tokenizer, label2id, args.max_length, mask_o_sources=mask_srcs)
    val_ds = align(load_jsonl(args.val), tokenizer, label2id, args.max_length) if args.val else None

    collator = DataCollatorForTokenClassification(tokenizer)

    def compute_metrics(p):
        from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
        preds = np.argmax(p.predictions, axis=2)
        true_lab, pred_lab = [], []
        for pred, lab in zip(preds, p.label_ids):
            tl = [id2label[l] for l in lab if l != -100]
            pl = [id2label[pr] for (pr, l) in zip(pred, lab) if l != -100]
            true_lab.append(tl)
            pred_lab.append(pl)
        return {
            "precision": precision_score(true_lab, pred_lab),
            "recall": recall_score(true_lab, pred_lab),
            "f1": f1_score(true_lab, pred_lab),
        }

    if args.grad_checkpointing:
        model.config.use_cache = False
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=args.grad_checkpointing,
        optim=args.optim,
        learning_rate=args.lr,
        eval_strategy="epoch" if val_ds else "no",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
        load_best_model_at_end=bool(val_ds),
        metric_for_best_model="f1",
        bf16=not args.no_bf16,
        report_to=[],
    )
    import torch
    import torch.nn.functional as F

    class WeightedKDTrainer(Trainer):
        """Trainer with (a) class-weighted CE (--o-weight) and (b) online logit
        distillation (--distill-from). KD's KL runs over every attended token --
        including weak-masked O positions the hard CE ignores -- so the teacher
        fills exactly the supervision hole that O-masking opens."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            C = logits.size(-1)
            w = torch.ones(C, device=logits.device)
            w[label2id["O"]] = args.o_weight
            flat_labels = labels.view(-1)
            if (flat_labels != -100).any():
                ce = F.cross_entropy(logits.view(-1, C).float(), flat_labels,
                                     weight=w, ignore_index=-100)
            else:
                ce = logits.sum() * 0.0  # keep graph; all-masked batch
            loss = ce
            if teacher is not None:
                if teacher.device != logits.device:
                    teacher.to(logits.device)
                with torch.no_grad():
                    tlogits = teacher(**inputs).logits
                mask = inputs["attention_mask"].bool()
                T = args.distill_temp
                kd = F.kl_div(F.log_softmax(logits[mask].float() / T, dim=-1),
                              F.softmax(tlogits[mask].float() / T, dim=-1),
                              reduction="batchmean") * T * T
                loss = (1 - args.distill_alpha) * ce + args.distill_alpha * kd
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedKDTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                                data_collator=collator, compute_metrics=compute_metrics if val_ds else None)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    sys.stderr.write(f"\nsaved model to {args.output_dir}\n")

    if args.test:
        test_ds = align(load_jsonl(args.test), tokenizer, label2id, args.max_length)
        sys.stderr.write(f"test metrics: {trainer.evaluate(test_ds)}\n")

    sys.stderr.write(f"\nExport to ONNX for nym:\n"
                     f"  scripts/convert_openmed_onnx.sh {args.output_dir} models/nym-pii-onnx\n"
                     f"then set [ner] token_model = \"/abs/path/to/models/nym-pii-onnx\"\n")


if __name__ == "__main__":
    main()
