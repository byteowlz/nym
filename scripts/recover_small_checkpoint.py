#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "transformers==5.13.0", "torch==2.14.1", "onnx==1.23.2",
#   "onnxruntime==1.30.0", "numpy==2.4.6", "safetensors==0.8.0",
# ]
# ///
"""Recover the pinned public SMALL v3 fp32 ONNX without initializing new weights.

    uv run scripts/recover_small_checkpoint.py --download
    uv run scripts/recover_small_checkpoint.py --self-test

Default source/output are outside git in /tmp/nym-model-study/train-recovery.
Only config.vocab_size is corrected (256000 -> actual pruned embedding rows).
The 81 BIO labels and tokenizer files are preserved exactly. Every state tensor
must have a unique, graph-proven source; linear weights are mapped by module node
names, NOT initializer numbers, shapes, or ordering. ONNX MatMul RHS is the
transpose of Torch Linear.weight, including square matrices. Unsupported folding,
missing/ambiguous tensors and unused initializers fail closed.

As in export_onnx.py, ONNX parity runs in a fresh torch-free subprocess. Only a
fully verified, reloaded checkpoint is moved into the output path. Use the default
SDPA backend: eager attention differs on fully masked local-attention padding
queries, even with these identical weights. The gate checks ALL logits, including
padding, and never relaxes its tolerance. This restores model weights, not
unavailable optimizer/scheduler/RNG state. Nothing is published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request

REVISION = "4348999cd3c2e20c49615e9af7c6bbb45b64cd85"
REPO = "Wismut/nym-pii-multilingual-small"
ROOT = Path("/tmp/nym-model-study/train-recovery")
HASHES = {
    "model.onnx": "60c2ae1a5992e43d4022a29dcc81cc45250bdb08fd2ee472cfb4d8800bbe87aa",
    "config.json": "3f07065571e22bb73eba28ddb1fae4509c0cf703762e3bfa7a6ab2111cf7cb88",
    "tokenizer.json": "c299144e68dfec1dc536204a7ae3712710c5c6cade9269a83f5042250d47d8de",
    "tokenizer_config.json": "02055b886266b1a475bc324da83e273fe96e4974002f9d970f26e94aa73e5885",
}
LENGTHS = (8, 16, 32, 48, 64, 96, 128, 192, 256, 320, 512)
TOL = 1e-3
TEXTS = (
    "The buffer size is 8192 bytes, not a PIN. My authentication PIN is 4826.",
    "My name is Alex Example. Email: alex@example.invalid. Date: 2026-07-18.",
    "Mein Name ist Erika Beispiel. Le mot de passe est fictif. El teléfono es privado.",
    "私の名前は田中です。测试文本。مرحبا بالعالم. Пример текста. नमस्ते।",
)


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_source(source, download):
    source.mkdir(parents=True, exist_ok=True)
    for name, expected in HASHES.items():
        path = source / name
        if not path.exists() and download:
            temp = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(
                f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}", temp
            )
            if sha256(temp) != expected:
                raise ValueError(f"download checksum mismatch: {name}")
            temp.rename(path)
        if not path.exists() or sha256(path) != expected:
            raise ValueError(f"missing or wrong pinned source: {path}; use --download")


class GraphWeights:
    """Resolve only explicit initializer/Constant, Identity and Transpose edges."""

    def __init__(self, graph):
        self.initializers = {}
        self.producers = {}
        self.nodes = {}
        for tensor in graph.initializer:
            if tensor.name in self.initializers:
                raise ValueError(f"duplicate initializer: {tensor.name}")
            self.initializers[tensor.name] = tensor
        for node in graph.node:
            if node.name in self.nodes:
                raise ValueError(f"duplicate node name: {node.name}")
            self.nodes[node.name] = node
            for output in node.output:
                if output in self.producers or output in self.initializers:
                    raise ValueError(f"ambiguous producer: {output}")
                self.producers[output] = node

    def resolve(self, name, seen=frozenset()):
        from onnx import helper, numpy_helper

        if name in seen:
            raise ValueError(f"cyclic weight edge: {name}")
        if name in self.initializers:
            return numpy_helper.to_array(self.initializers[name]), name, []
        node = self.producers.get(name)
        if node is None:
            raise ValueError(f"missing weight value: {name}")
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type == "Constant" and set(attrs) == {"value"}:
            return numpy_helper.to_array(attrs["value"]), name, ["Constant"]
        if node.op_type not in ("Identity", "Transpose") or len(node.input) != 1:
            raise ValueError(f"unsupported weight folding: {node.name} ({node.op_type})")
        value, origin, transforms = self.resolve(node.input[0], seen | {name})
        if node.op_type == "Transpose":
            perm = attrs.get("perm", tuple(reversed(range(value.ndim))))
            value = value.transpose(perm)
            transforms = transforms + [f"Transpose{list(perm)}"]
        return value, origin, transforms

    def tensor(self, key):
        """Known Torch module paths to exact legacy-export operation paths."""
        module, kind = key.rsplit(".", 1)
        path = "/m/" + module.replace(".", "/")
        # Legacy exporter encodes the ModuleList index as 'layers.0'.
        import re

        path = re.sub(r"/layers/(\d+)/", r"/layers.\1/", path)
        linear = kind == "weight" and (
            module.endswith((".Wqkv", ".Wo", ".Wi", ".dense")) or module == "classifier"
        )
        if linear:
            node = self.nodes.get(path + "/MatMul")
            if node is None or node.op_type != "MatMul" or len(node.input) != 2:
                raise ValueError(f"missing explicit linear MatMul for {key}")
            value, origin, transforms = self.resolve(node.input[1])
            if value.ndim != 2:
                raise ValueError(f"non-matrix RHS for {key}")
            return value.T, origin, transforms + ["MatMul RHS -> Linear transpose"], node.name
        name = "m." + key
        value, origin, transforms = self.resolve(name)
        # Named weights still need a matching operation, not just a convenient shape.
        if module.endswith("tok_embeddings"):
            operation, slot = "Gather", 0
        elif kind == "weight" and module.endswith("norm"):
            operation, slot = "LayerNormalization", 1
        elif key == "classifier.bias":
            operation, slot = "Add", 0
        else:
            raise ValueError(f"unsupported state tensor: {key}")
        node = self.nodes.get(path + "/" + operation)
        if node is None or node.op_type != operation or node.input[slot] != name:
            raise ValueError(f"missing named weight consumer for {key}")
        return value, origin, transforms, node.name


def recover_state(graph, expected):
    """Return strict, bijective fp32 state and a per-tensor audit manifest."""
    import numpy as np
    import torch

    weights = GraphWeights(graph)
    state, audit, used = {}, {}, set()
    errors = []
    for key, target in expected.items():
        try:
            value, origin, transforms, node = weights.tensor(key)
            if origin in used:
                raise ValueError(f"shared/ambiguous source {origin}")
            if value.shape != tuple(target.shape):
                raise ValueError(f"shape {value.shape} != {tuple(target.shape)}")
            if value.dtype != np.float32 or target.dtype != torch.float32:
                raise ValueError(f"not fp32: {value.dtype}, {target.dtype}")
            if not np.isfinite(value).all():
                raise ValueError("non-finite source")
            state[key] = torch.from_numpy(value.copy())
            used.add(origin)
            audit[key] = {
                "source": origin, "node": node, "transforms": transforms,
                "shape": list(value.shape),
                "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            }
        except ValueError as error:
            errors.append(f"{key}: {error}")
    unused = sorted(set(weights.initializers) - used)
    if unused:
        errors.append(f"unmapped initializers: {unused}")
    if errors:
        raise ValueError("weight recovery rejected:\n" + "\n".join(errors))
    return state, audit


def validate_labels(config):
    labels = config["id2label"]
    if set(labels) != {str(i) for i in range(81)} or labels["0"] != "O":
        raise ValueError("expected the full 81-class BIO map")
    for i in range(1, 81, 2):
        if not labels[str(i)].startswith("B-") or labels[str(i + 1)] != "I-" + labels[str(i)][2:]:
            raise ValueError(f"invalid BIO pair at {i}")
    if len(set(labels.values())) != 81 or config["label2id"] != {label: int(i) for i, label in labels.items()}:
        raise ValueError("label maps are not bijective")


def tokenizer_proof(source, checkpoint, vocab_size):
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    import numpy as np

    original = Tokenizer.from_file(str(source / "tokenizer.json"))
    recovered = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    vocab = original.get_vocab(with_added_tokens=True)
    if len(vocab) != vocab_size or set(vocab.values()) != set(range(vocab_size)):
        raise ValueError("pruned tokenizer IDs do not exactly cover embedding rows")
    if recovered.get_vocab() != vocab:
        raise ValueError("reloaded tokenizer vocabulary changed")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if sha256(source / name) != sha256(checkpoint / name):
            raise ValueError(f"tokenizer file changed: {name}")
    cases = {}
    for i, text in enumerate(TEXTS):
        ids = original.encode(text).ids
        loaded = recovered(text, truncation=False)["input_ids"]
        if ids != loaded:
            raise ValueError(f"token IDs differ on fixture {i}")
        cases[f"text_{i}"] = (np.array([ids], dtype=np.int64), np.ones((1, len(ids)), np.int64))
    return recovered, cases


def parity_stage(npz_path, onnx_path, report_path, threads):
    # No imports of torch/transformers in this fresh process: OpenMP isolation.
    import numpy as np
    import onnxruntime as ort

    if "torch" in sys.modules:
        raise RuntimeError("ONNX worker imported torch")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    results = []
    with np.load(npz_path) as data:
        for name in sorted(k[:-4] for k in data.files if k.endswith("_ids")):
            expected = data[name + "_logits"]
            got = session.run(None, {"input_ids": data[name + "_ids"],
                                      "attention_mask": data[name + "_mask"]})[0]
            if got.shape != expected.shape or not np.isfinite(got).all():
                raise ValueError(f"invalid ONNX output for {name}")
            diff = float(np.abs(got - expected).max())
            attended = data[name + "_mask"].astype(bool)
            attended_diff = float(np.abs(got[attended] - expected[attended]).max())
            same = bool(np.array_equal(got.argmax(-1), expected.argmax(-1)))
            result = {"case": name, "shape": list(got.shape), "max_abs_diff": diff,
                      "attended_max_abs_diff": attended_diff,
                      "argmax_equal": same, "passed": diff < TOL and same}
            results.append(result)
            print(json.dumps(result), flush=True)
    report = {"torch_imported": "torch" in sys.modules, "tolerance": TOL, "cases": results}
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    if not all(r["passed"] for r in results) or report["torch_imported"]:
        raise ValueError("Torch/ONNX parity failed; no checkpoint promoted")


def verify_checkpoint(source, checkpoint, work, threads):
    import numpy as np
    import torch
    from transformers import AutoModelForTokenClassification

    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint, local_files_only=True, attn_implementation="sdpa"
    ).cpu().eval()
    if model.config._attn_implementation != "sdpa":
        raise ValueError("expected the training/export SDPA attention backend")
    tokenizer, cases = tokenizer_proof(source, checkpoint, model.config.vocab_size)
    rng = np.random.default_rng(0)
    for length in LENGTHS:
        ids = rng.integers(5, model.config.vocab_size, (1, length), dtype=np.int64)
        cases[f"random_{length}"] = (ids, np.ones_like(ids))
    # Unequal attended lengths, different IDs, right and left padding.
    for side in ("right", "left"):
        enc = tokenizer([TEXTS[0], TEXTS[2] * 12], padding="max_length", max_length=384,
                        truncation=True, return_tensors="np", padding_side=side)
        cases[f"padded_{side}"] = (enc["input_ids"], enc["attention_mask"])
    dump = {}
    with torch.no_grad():
        for name, (ids, mask) in cases.items():
            logits = model(input_ids=torch.from_numpy(ids), attention_mask=torch.from_numpy(mask)).logits
            if not torch.isfinite(logits).all():
                raise ValueError(f"non-finite Torch output: {name}")
            dump.update({name + "_ids": ids, name + "_mask": mask, name + "_logits": logits.numpy()})
    npz_path = work / "parity-inputs.npz"
    np.savez(npz_path, **dump)
    report_path = work / "parity.json"
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--parity-stage",
                    str(npz_path), str(source / "model.onnx"), str(report_path), str(threads)],
                   check=True, env={**os.environ, "OMP_NUM_THREADS": str(threads)})
    # A loss/backward smoke test, no optimizer step and no checkpoint mutation.
    model.train()
    ids, mask = cases["text_0"]
    ids, mask = torch.from_numpy(ids), torch.from_numpy(mask)
    labels = torch.zeros_like(ids)
    labels[:, 0] = 53  # synthetic BIO target; exercise the unchanged classifier.
    loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
    loss.backward()
    missing = [k for k, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    if not torch.isfinite(loss) or missing:
        raise ValueError(f"training backward failed: {missing}")
    return {"parity": json.loads(report_path.read_text()), "backward_loss": loss.item(),
            "attention_backend": model.config._attn_implementation,
            "finite_gradient_tensors": len(list(model.parameters())),
            "tokenizer_vocab": model.config.vocab_size, "token_ids_equal": True,
            "tokenizer_files_byte_identical": True}


def recover(source, output, download, threads):
    import onnx
    import torch
    import transformers
    from importlib.metadata import version
    from transformers import AutoConfig, AutoModelForTokenClassification
    from transformers.initialization import no_init_weights

    if transformers.__version__ != "5.13.0":
        raise ValueError("recovery requires transformers==5.13.0, the training forward version")
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    torch.set_num_threads(threads)
    prepare_source(source, download)
    raw_config = json.loads((source / "config.json").read_text())
    validate_labels(raw_config)
    graph = onnx.load(source / "model.onnx").graph
    embedding = GraphWeights(graph).initializers["m.model.embeddings.tok_embeddings.weight"]
    vocab_size, hidden = embedding.dims
    if hidden != 384 or raw_config["num_hidden_layers"] != 16:
        raise ValueError("not the authorized SMALL architecture")
    config = AutoConfig.from_pretrained(source, local_files_only=True)
    config.vocab_size = vocab_size
    # Suppress parameter initialization, then overwrite EVERY tensor strictly.
    # CPU construction still creates deterministic nonpersistent rotary buffers.
    with no_init_weights():
        model = AutoModelForTokenClassification.from_config(config, attn_implementation="sdpa")
    state, audit = recover_state(graph, model.state_dict())
    model.load_state_dict(state, strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".recovery-", dir=output.parent) as staging:
        checkpoint = Path(staging) / "checkpoint"
        model.save_pretrained(checkpoint)
        # HF serialization drops legacy metadata such as use_cache=False. Preserve
        # the public config verbatim in meaning, except the proven vocab correction.
        corrected_config = {**raw_config, "vocab_size": vocab_size}
        (checkpoint / "config.json").write_text(json.dumps(corrected_config, indent=2) + "\n")
        for name in ("tokenizer.json", "tokenizer_config.json"):
            shutil.copyfile(source / name, checkpoint / name)
        saved_config = json.loads((checkpoint / "config.json").read_text())
        if saved_config != corrected_config:
            raise ValueError("saved config changed beyond vocab_size")
        reloaded = AutoModelForTokenClassification.from_pretrained(checkpoint, local_files_only=True)
        loaded = reloaded.state_dict()
        if loaded.keys() != state.keys() or any(not torch.equal(loaded[k], state[k]) for k in state):
            raise ValueError("checkpoint save/reload changed weights")
        del reloaded, loaded
        proof = verify_checkpoint(source, checkpoint, output.parent, threads)
        report = {
            "repo": REPO, "revision": REVISION, "source_sha256": HASHES,
            "versions": {name: version(name) for name in (
                "transformers", "torch", "onnx", "onnxruntime", "numpy", "tokenizers", "safetensors")},
            "script_sha256": sha256(Path(__file__)),
            "config_only_vocab_size_changed": True,
            "source_vocab_size": raw_config["vocab_size"], "recovered_vocab_size": vocab_size,
            "state_tensors": len(state), "parameters": sum(v.numel() for v in state.values()),
            "full_state_matched": True, "save_reload_weights_exact": True,
            "labels": raw_config["id2label"], "mapping": audit, "verification": proof,
            "artifact_sha256": {p.name: sha256(p) for p in checkpoint.iterdir() if p.is_file()},
        }
        (checkpoint / "recovery-report.json").write_text(json.dumps(report, indent=2) + "\n")
        checkpoint.rename(output)
    print(f"Verified trainable checkpoint: {output}")


def self_test():
    """Offline, synthetic graph tests; no downloads or real model required."""
    import unittest
    import numpy as np
    import torch
    from onnx import helper, numpy_helper

    class MappingTests(unittest.TestCase):
        def graph(self, extra=(), nodes=()):
            value = np.arange(9, dtype=np.float32).reshape(3, 3)
            initializer = numpy_helper.from_array(value, "folded")
            node = helper.make_node("MatMul", ["x", "folded"], ["y"], name="/m/head/dense/MatMul")
            return helper.make_graph([node, *nodes], "test", [], [], [initializer, *extra]), value

        def test_square_linear_transposed_by_node_not_shape(self):
            graph, value = self.graph()
            state, audit = recover_state(graph, {"head.dense.weight": torch.empty(3, 3)})
            self.assertTrue(torch.equal(state["head.dense.weight"], torch.from_numpy(value.T.copy())))
            self.assertEqual(audit["head.dense.weight"]["source"], "folded")

        def test_missing_and_unused_rejected(self):
            graph, _ = self.graph()
            with self.assertRaisesRegex(ValueError, "missing explicit linear.*classifier.weight"):
                recover_state(graph, {"classifier.weight": torch.empty(3, 3)})
            with self.assertRaisesRegex(ValueError, "unmapped initializers"):
                recover_state(graph, {})

        def test_shape_and_dtype_rejected(self):
            graph, _ = self.graph()
            with self.assertRaisesRegex(ValueError, "shape"):
                recover_state(graph, {"head.dense.weight": torch.empty(4, 3)})
            graph.initializer[0].CopyFrom(numpy_helper.from_array(np.ones((3, 3), np.float16), "folded"))
            with self.assertRaisesRegex(ValueError, "not fp32"):
                recover_state(graph, {"head.dense.weight": torch.empty(3, 3)})

        def test_duplicate_name_and_source_rejected(self):
            graph, _ = self.graph()
            graph.initializer.add().CopyFrom(graph.initializer[0])
            with self.assertRaisesRegex(ValueError, "duplicate initializer"):
                GraphWeights(graph)
            graph, _ = self.graph(nodes=[helper.make_node(
                "MatMul", ["x", "folded"], ["z"], name="/m/classifier/MatMul")])
            with self.assertRaisesRegex(ValueError, "ambiguous source"):
                recover_state(graph, {"head.dense.weight": torch.empty(3, 3),
                                      "classifier.weight": torch.empty(3, 3)})

        def test_constant_transpose_identity_resolved(self):
            value = np.arange(6, dtype=np.float32).reshape(2, 3)
            graph = helper.make_graph([
                helper.make_node("Constant", [], ["c"], name="constant",
                                 value=numpy_helper.from_array(value)),
                helper.make_node("Transpose", ["c"], ["t"], name="transpose", perm=[1, 0]),
                helper.make_node("Identity", ["t"], ["i"], name="identity"),
            ], "test", [], [])
            got, source, transforms = GraphWeights(graph).resolve("i")
            np.testing.assert_array_equal(got, value.T)
            self.assertEqual((source, transforms), ("c", ["Constant", "Transpose[1, 0]"]))

        def test_unsupported_folding_and_nonfinite_rejected(self):
            graph, _ = self.graph()
            graph.initializer[0].CopyFrom(numpy_helper.from_array(np.full((3, 3), np.nan, np.float32), "folded"))
            with self.assertRaisesRegex(ValueError, "non-finite"):
                recover_state(graph, {"head.dense.weight": torch.empty(3, 3)})
            graph = helper.make_graph([helper.make_node("Mul", ["a", "b"], ["c"], name="bad")], "test", [], [])
            with self.assertRaisesRegex(ValueError, "unsupported weight folding"):
                GraphWeights(graph).resolve("c")

        def test_full_label_map_required(self):
            with self.assertRaisesRegex(ValueError, "81-class"):
                validate_labels({"id2label": {"0": "O"}, "label2id": {"O": 0}})

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MappingTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=ROOT / "source")
    parser.add_argument("--output", type=Path, default=ROOT / "small-v3-recovered")
    parser.add_argument("--download", action="store_true", help="download missing pinned public files only")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--self-test", action="store_true", help="offline synthetic graph regression tests")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.self_test:
        self_test()
    else:
        recover(args.source.resolve(), args.output.resolve(), args.download, args.threads)


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "--parity-stage":
        parity_stage(*sys.argv[2:5], int(sys.argv[5]))
    else:
        main()
