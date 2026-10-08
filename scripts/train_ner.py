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

Continuation from a trained checkpoint (never rebuild or reset its label head):
  --base-model /local/checkpoint --frozen-label-config /local/checkpoint/config.json
The supplied BIO map is authoritative even for a corpus subset. Unknown labels,
changed checkpoint IDs, and missing/mismatched learned tensors fail closed.

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

# Also support existing importlib callers outside the scripts directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))


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


def validate_labels(rows, label2id):
    """Reject unknown entity types even when truncation would hide their tokens."""
    unknown = sorted({e["label"] for r in rows for e in r["entities"]
                      if any(f"{p}-{e['label']}" not in label2id for p in ("B", "I"))})
    if unknown:
        raise ValueError(f"unmapped corpus labels in frozen taxonomy: {unknown}")


def build_labels(rows, frozen_label_config=None):
    """Derive BIO IDs as before, or preserve an explicit checkpoint map exactly.

    Frozen mode validates contiguous, bijective, complete BIO maps and every
    corpus entity. It never sorts, shrinks, adds labels or mutates the config.
    """
    if frozen_label_config is None:
        types = sorted({e["label"] for r in rows for e in r["entities"]})
        labels = ["O"] + [f"{p}-{t}" for t in types for p in ("B", "I")]
        return labels, {l: i for i, l in enumerate(labels)}
    config = json.loads(Path(frozen_label_config).read_text(encoding="utf-8"))
    mapping = config.get("id2label")
    if not isinstance(mapping, dict) or not mapping or set(mapping) != {str(i) for i in range(len(mapping))}:
        raise ValueError("frozen id2label must have contiguous IDs starting at zero")
    labels = [mapping[str(i)] for i in range(len(mapping))]
    if not all(isinstance(label, str) for label in labels):
        raise ValueError("frozen labels must be strings")
    types = {label[2:] for label in labels if label.startswith("B-") and label[2:]}
    expected = {"O"} | {f"{p}-{t}" for t in types for p in ("B", "I")}
    label2id = {label: i for i, label in enumerate(labels)}
    if len(label2id) != len(labels) or set(labels) != expected or config.get("label2id") != label2id:
        raise ValueError("frozen BIO maps must be complete, unique and inverse")
    validate_labels(rows, label2id)
    return labels, label2id


def validate_checkpoint_labels(config, label2id):
    """Fail on same-size ID permutation as well as a differently sized head."""
    if config.label2id != label2id or config.id2label != {i: l for l, i in label2id.items()}:
        raise ValueError("base checkpoint label IDs differ from frozen taxonomy; refusing head reset")


def load_training_model(model_cls, base_model, labels, label2id, frozen=False):
    """Frozen continuation requires every learned tensor, not default init."""
    kwargs = dict(num_labels=len(labels), id2label={i: l for l, i in label2id.items()},
                  label2id=label2id, ignore_mismatched_sizes=not frozen)
    if not frozen:
        return model_cls.from_pretrained(base_model, **kwargs)
    # Do not override config IDs in frozen mode: that would hide a same-size
    # permutation before validate_checkpoint_labels could detect it.
    model, info = model_cls.from_pretrained(base_model, output_loading_info=True,
                                          ignore_mismatched_sizes=False)
    problems = {key: info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                if info.get(key)}
    if problems:
        raise ValueError(f"frozen continuation requires exact learned tensors: {problems}")
    validate_checkpoint_labels(model.config, label2id)
    return model


def align(rows, tokenizer, label2id, max_length, mask_o_sources=frozenset()):
    """Tokenize each example and assign a label id per token from char spans.
    First token of an entity -> B-, subsequent overlapping tokens -> I-, tokens
    outside any entity -> O, special/pad tokens -> -100 (ignored in the loss).

    mask_o_sources: `source` values with WEAK labels (e.g. teacher-labeled real
    text, where an undetected entity would otherwise train as a false "O").
    For those records, O tokens become -100 — only positive spans supervise."""
    from gold_ner import align_record
    validate_labels(rows, label2id)
    return [align_record(row, tokenizer, label2id, max_length,
                         weak=row.get("source") in mask_o_sources) for row in rows]


def load_training_data(args):
    """Resolve legacy or complete-gold inputs before downloading any model assets."""
    bundle = None
    if args.gold_bundle:
        if args.val or args.test:
            raise ValueError("gold bundle already declares selection/holdout")
        from gold_ner import load_bundle
        bundle = load_bundle(args.gold_bundle)
        parts = {s: bundle[s] for s in ("train", "selection", "holdout")}
    else:
        parts = {"train": load_jsonl(args.train),
                 "selection": load_jsonl(args.val) if args.val else None,
                 "holdout": load_jsonl(args.test) if args.test else None}
    require_gold = args.require_gold or bundle is not None or any(
        r.get("schema") == "nym.ner.gold.v1" for rows in parts.values() for r in (rows or []))
    if require_gold:
        from gold_ner import validate_partitions
        validate_partitions(parts)
        if args.mask_o_sources or args.distill_from:
            raise ValueError("gold mode forbids weak-negative promotion and unmasked distillation")
    if args.distill_from and any(r.get("masked_spans") for rows in parts.values() for r in (rows or [])):
        raise ValueError("distillation cannot supervise unknown annotation regions")
    if args.allow_head_reset and args.frozen_label_config:
        raise ValueError("frozen continuation never permits head reset")
    return parts["train"], parts["selection"], parts["holdout"], bundle, require_gold


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = ap.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--train", nargs="+", help="JSONL path(s)/glob(s)")
    inputs.add_argument("--gold-bundle", type=Path, help="validated complete-gold source-split bundle")
    ap.add_argument("--require-gold", action="store_true", help="reject incomplete or unbound training annotations")
    ap.add_argument("--allow-head-reset", action="store_true", help="explicitly authorize a new classifier taxonomy in gold mode")
    ap.add_argument("--val", nargs="+", default=None)
    ap.add_argument("--test", nargs="+", default=None)
    ap.add_argument("--base-model", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--frozen-label-config", type=Path,
                    help="checkpoint config.json: preserve all BIO IDs and learned head; "
                         "fail on unknown corpus labels, changed IDs or incomplete weights")
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

    train_rows, val_rows, test_rows, bundle, require_gold = load_training_data(args)
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
    taxonomy_rows = train_rows
    if bundle:
        # Types are operator-declared before splitting, not inferred from holdout examples.
        taxonomy_rows = train_rows + [{"entities": [{"label": label} for label in bundle["label_types"]]}]
    labels, label2id = build_labels(taxonomy_rows, args.frozen_label_config)
    for rows in (val_rows, test_rows):
        if rows is not None:
            validate_labels(rows, label2id)
    from transformers import AutoConfig, AutoTokenizer
    if args.frozen_label_config or (require_gold and not args.allow_head_reset):
        validate_checkpoint_labels(AutoConfig.from_pretrained(args.base_model), label2id)
    id2label = {i: l for l, i in label2id.items()}
    sys.stderr.write(f"{len(labels)} BIO labels over {(len(labels)-1)//2} entity types\n")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    if args.dry_run:
        sample = align(train_rows[:200], tokenizer, label2id, args.max_length)
        if require_gold:
            from gold_ner import supervised_training
            supervised = supervised_training(sample, label2id["O"])
            sys.stderr.write(f"gold alignment: {len(supervised)} supervised units; no private token dump\n")
            return
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

    from transformers import AutoConfig

    model_cls = AutoModelForTokenClassification
    if AutoConfig.from_pretrained(args.base_model).__class__.__name__.startswith("Gemma3"):
        # transformers 5.13 has no Gemma3 token-classification head, and
        # Auto.register() proved unreliable for it -- load the class directly.
        sys.path.insert(0, str(Path(__file__).parent))
        from gemma3_tc import Gemma3ForTokenClassification
        model_cls = Gemma3ForTokenClassification

    model = load_training_model(model_cls, args.base_model, labels, label2id,
                                frozen=bool(args.frozen_label_config))

    teacher = None
    if args.distill_from:
        teacher = AutoModelForTokenClassification.from_pretrained(args.distill_from).eval()
        if teacher.config.label2id != label2id:
            sys.exit(f"teacher label map differs from student's ({args.distill_from}); "
                     f"KD logits would supervise the wrong classes")
        for p in teacher.parameters():
            p.requires_grad_(False)

    train_ds = align(train_rows, tokenizer, label2id, args.max_length, mask_o_sources=mask_srcs)
    val_ds = align(val_rows, tokenizer, label2id, args.max_length) if val_rows is not None else None
    if require_gold:
        from gold_ner import supervised_training, supervised_selection
        before = len(train_ds)
        train_ds = supervised_training(train_ds, label2id["O"])
        val_ds = supervised_selection(val_ds)
        sys.stderr.write(f"excluded {before - len(train_ds)} fully masked optimizer units\n")

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
        test_ds = align(test_rows, tokenizer, label2id, args.max_length)
        sys.stderr.write(f"test metrics: {trainer.evaluate(test_ds)}\n")

    sys.stderr.write(f"\nExport to ONNX for nym:\n"
                     f"  scripts/convert_openmed_onnx.sh {args.output_dir} models/nym-pii-onnx\n"
                     f"then set [ner] token_model = \"/abs/path/to/models/nym-pii-onnx\"\n")


if __name__ == "__main__":
    main()
