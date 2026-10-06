#!/usr/bin/env python3
"""Offline CPU checkpoint/export proof, with Torch and ORT in separate processes.

reference --checkpoint DIR --source DIR --work DIR [--rows CORPUS.jsonl ...]
compare --reference DIR --model DIR --report FILE
quantize --model DIR --output DIR

References cover dynamic lengths 8..512, high vocabulary IDs, real text and
left/right padded unequal batches. Comparison checks finite full logits, softmax
probabilities, argmax and fixed-0.5 token decisions, separately reporting attended
and masked positions. Quantized tolerances must be explicitly supplied; they do
not waive ANY decision disagreement. This is numerical proof, not a recall gate.
All artifacts stay outside git; no download, config mutation or publication.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys

LENGTHS = (8, 16, 32, 48, 64, 96, 128, 192, 256, 320, 512)
TEXTS = (
    "Compile bash with a 4096-byte buffer; not a PIN. Public docs: https://docs.rs/serde/.",
    "Patient Mira Linden was born on 1986-04-19; her bank PIN is 5729.",
    "患者姓名：林明；出生日期：1986-04-19。银行卡密码：5729。",
    "Пациент Анна Орлова; дата рождения 1986-04-19, ПИН банковской карты 5729.",
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest(directory):
    return {p.name: sha(p) for p in sorted(Path(directory).iterdir()) if p.is_file()}


def tokenizer_fingerprint(directory):
    tokenizer = json.loads((Path(directory) / "tokenizer.json").read_text())
    tokenizer.update(truncation=None, padding=None)
    return hashlib.sha256(json.dumps(tokenizer, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def private(path):
    path = Path(path).expanduser().resolve()
    if any((parent / ".git").exists() for parent in (path, *path.parents)):
        raise ValueError("proof artifacts must stay outside git worktrees")
    return path


def write_json(path, value):
    private(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def validate_taxonomy(config, source):
    labels = config.get("id2label", {})
    expected = source.get("id2label", {})
    if (labels != expected or config.get("label2id") != source.get("label2id")
            or len(labels) != 81 or labels.get("0") != "O"
            or set(labels) != {str(i) for i in range(81)}
            or config.get("label2id") != {label: int(i) for i, label in labels.items()}):
        raise ValueError("frozen original 81 BIO taxonomy changed")


def input_cases(tokenizer, vocab, pad, texts):
    import numpy as np

    tokenizer.no_truncation()
    tokenizer.no_padding()
    rng = np.random.default_rng(47)
    cases = {}
    for length in LENGTHS:
        ids = rng.integers(5, vocab, (1, length), dtype=np.int64)
        ids[0, -1] = vocab - 1  # actual learned embedding bound, not 256k config lore
        cases[f"random_{length}"] = (ids, np.ones_like(ids))
    for i, text in enumerate(texts):
        encoded = tokenizer.encode(text)
        if len(encoded.ids) > 512:
            raise ValueError("proof row would be truncated")
        ids = np.array([encoded.ids], dtype=np.int64)
        cases[f"text_{i}"] = (ids, np.ones_like(ids))
    for side in ("left", "right"):
        ids = np.full((2, 384), pad, dtype=np.int64)
        mask = np.zeros_like(ids)
        for row, length in enumerate((73, 301)):
            start = 384 - length if side == "left" else 0
            ids[row, start:start + length] = rng.integers(5, vocab, length)
            mask[row, start:start + length] = 1
        cases[f"padded_{side}"] = (ids, mask)
    return cases


def reference(args):
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModelForTokenClassification

    if version("transformers") != args.transformers_version:
        raise ValueError("reference must use the training transformers version")
    work = private(args.work)
    work.mkdir(mode=0o700, parents=True, exist_ok=False)
    source = json.loads((args.source / "config.json").read_text())
    config = json.loads((args.checkpoint / "config.json").read_text())
    validate_taxonomy(config, source)
    before = manifest(args.checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(
        args.checkpoint, local_files_only=True, attn_implementation="sdpa").cpu().eval()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    vocab = model.get_input_embeddings().num_embeddings
    if vocab != config["vocab_size"] or max(tokenizer.get_vocab().values()) + 1 != vocab:
        raise ValueError("actual embedding/config/tokenizer vocabulary disagreement")
    texts = list(TEXTS)
    for path in args.rows:
        texts += [json.loads(line)["text"] for line in path.read_text().splitlines() if line.strip()]
    dump = {}
    with torch.inference_mode():
        for name, (ids, mask) in input_cases(tokenizer, vocab, config.get("pad_token_id") or 0, texts).items():
            logits = model(input_ids=torch.from_numpy(ids), attention_mask=torch.from_numpy(mask)).logits.numpy()
            if not np.isfinite(logits).all():
                raise ValueError("nonfinite reference")
            dump.update({name + "_ids": ids, name + "_mask": mask, name + "_logits": logits})
    np.savez(work / "inputs.npz", **dump)
    if manifest(args.checkpoint) != before:
        raise ValueError("checkpoint changed during reference")
    write_json(work / "reference.json", {
        "device": "cpu", "attention": model.config._attn_implementation, "threshold": 0.5,
        "threads": args.threads, "actual_vocab": vocab,
        "tokenizer_semantic_sha256": tokenizer_fingerprint(args.checkpoint), "id2label": config["id2label"],
        "label2id": config["label2id"], "checkpoint_sha256": before,
        "source_sha256": manifest(args.source), "inputs_sha256": sha(work / "inputs.npz"),
        "script_sha256": sha(__file__), "corpus_sha256": {str(p): sha(p) for p in args.rows},
        "versions": {name: version(name) for name in ("torch", "transformers", "numpy", "tokenizers")},
        "onnxruntime_imported": "onnxruntime" in sys.modules,
    })
    if "onnxruntime" in sys.modules:
        raise ValueError("reference process imported ORT")


def softmax(logits):
    import numpy as np

    values = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


def compare_arrays(expected, actual, mask, logit_atol, probability_atol):
    import numpy as np

    if (expected.shape != actual.shape or expected.ndim != 3 or mask.shape != expected.shape[:2]
            or not np.isfinite(expected).all() or not np.isfinite(actual).all()):
        raise ValueError("invalid/nonfinite logits or mask shape")
    ep, ap = softmax(expected), softmax(actual)
    ei, ai = ep.argmax(-1), ap.argmax(-1)
    ed = np.where(ep.max(-1) >= 0.5, ei, 0)
    ad = np.where(ap.max(-1) >= 0.5, ai, 0)
    result = {"max_logit_error": float(np.abs(expected - actual).max()),
              "max_probability_error": float(np.abs(ep - ap).max())}
    for name, select in (("all", np.ones_like(mask, dtype=bool)),
                         ("attended", mask.astype(bool)), ("masked", ~mask.astype(bool))):
        result[name] = {"positions": int(select.sum()),
                        "argmax_disagreements": int(((ei != ai) & select).sum()),
                        "decision_disagreements": int(((ed != ad) & select).sum())}
    result["passed"] = (result["max_logit_error"] <= logit_atol
                        and result["max_probability_error"] <= probability_atol
                        and result["all"]["argmax_disagreements"] == 0
                        and result["all"]["decision_disagreements"] == 0)
    return result


def compare(args):
    import numpy as np
    import onnxruntime as ort

    if "torch" in sys.modules:
        raise ValueError("comparison process imported Torch")
    metadata = json.loads((args.reference / "reference.json").read_text())
    if sha(args.reference / "inputs.npz") != metadata["inputs_sha256"]:
        raise ValueError("reference inputs changed")
    config = json.loads((args.model / "config.json").read_text())
    validate_taxonomy(config, metadata)
    import onnx
    from tokenizers import Tokenizer

    model_file = args.model / ("model_int8.onnx" if (args.model / "model_int8.onnx").exists() else "model.onnx")
    graph = onnx.load(model_file, load_external_data=False).graph
    embeddings = [i for i in graph.initializer if "tok_embeddings.weight" in i.name and len(i.dims) == 2]
    if len(embeddings) != 1 or embeddings[0].dims[0] != metadata["actual_vocab"]:
        raise ValueError("actual exported embedding vocabulary changed")
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    if (max(tokenizer.get_vocab().values()) + 1 != metadata["actual_vocab"]
            or tokenizer_fingerprint(args.model) != metadata["tokenizer_semantic_sha256"]):
        raise ValueError("export tokenizer changed beyond truncation/padding")
    before = manifest(args.model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_file), options, providers=["CPUExecutionProvider"])
    results = {}
    with np.load(args.reference / "inputs.npz", allow_pickle=False) as data:
        for key in data.files:
            if not key.endswith("_ids"):
                continue
            name = key[:-4]
            actual = session.run(None, {"input_ids": data[key], "attention_mask": data[name + "_mask"]})[0]
            results[name] = compare_arrays(data[name + "_logits"], actual, data[name + "_mask"],
                                           args.logit_atol, args.probability_atol)
    if not results or manifest(args.model) != before:
        raise ValueError("empty proof or model changed during comparison")
    report = {"passed": all(row["passed"] for row in results.values()), "device": "cpu",
              "threshold": 0.5, "torch_imported": "torch" in sys.modules,
              "actual_vocab": metadata["actual_vocab"], "export_config_vocab": config["vocab_size"],
              "logit_atol": args.logit_atol, "probability_atol": args.probability_atol,
              "export_sha256": before, "reference_sha256": sha(args.reference / "reference.json"),
              "script_sha256": sha(__file__), "onnxruntime": version("onnxruntime"), "cases": results}
    write_json(args.report, report)
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}))
    return int(not report["passed"])


def quantize(args):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    output = private(args.output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    before = manifest(args.model)
    quantize_dynamic(str(args.model / "model.onnx"), str(output / "model.onnx"),
                     weight_type=QuantType.QInt8, per_channel=True, reduce_range=False,
                     op_types_to_quantize=["MatMul", "Gemm"])
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if (args.model / name).exists():
            shutil.copyfile(args.model / name, output / name)
    if manifest(args.model) != before:
        raise ValueError("source export changed")
    write_json(output / "quantization.json", {
        "source_sha256": before, "output_sha256": manifest(output), "onnxruntime": version("onnxruntime"),
        "weight_type": "QInt8", "per_channel": True, "reduce_range": False,
        "operators": ["MatMul", "Gemm"], "device": "cpu", "vocab_pruned": False,
    })


def main():
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest="stage", required=True)
    ref = subs.add_parser("reference")
    ref.add_argument("--checkpoint", type=Path, required=True)
    ref.add_argument("--source", type=Path, required=True)
    ref.add_argument("--work", type=Path, required=True)
    ref.add_argument("--rows", type=Path, nargs="*", default=[])
    ref.add_argument("--transformers-version", default="5.13.0")
    ref.add_argument("--threads", type=int, default=2)
    comp = subs.add_parser("compare")
    comp.add_argument("--reference", type=Path, required=True)
    comp.add_argument("--model", type=Path, required=True)
    comp.add_argument("--report", type=Path, required=True)
    comp.add_argument("--threads", type=int, default=2)
    comp.add_argument("--logit-atol", type=float, default=1e-3)
    comp.add_argument("--probability-atol", type=float, default=1e-4)
    quant = subs.add_parser("quantize")
    quant.add_argument("--model", type=Path, required=True)
    quant.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if hasattr(args, "threads") and args.threads < 1:
        parser.error("threads must be positive")
    if args.stage == "compare":
        import math
        if any(not math.isfinite(v) or v < 0 for v in (args.logit_atol, args.probability_atol)):
            parser.error("tolerances must be finite and nonnegative")
    return {"reference": reference, "compare": compare, "quantize": quantize}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
