#!/usr/bin/env python3
"""OOD eval for PII token-classification checkpoints (torch or onnxruntime).

Reads HF checkpoints directly, so architectures with their own tokenizers
(mDeBERTa, XLM-R) can be compared against mmBERT without an ONNX/vocab-prune
round trip. `--backend onnx` runs the same metrics and the same span decoding
over an exported model.onnx, so a torch-vs-ORT delta is attributable to the
runtime rather than to two harnesses that merely resemble each other.

ALWAYS evaluate a known-value control alongside the candidates (`final2` = the
shipped v2: torch/unpruned/len512 = ai4 62.0, non-Latin 69.6). If the control
does not reproduce, the harness is wrong and every other number in the run is
meaningless -- that check is what caught a bogus int8 "100% agreement" and a
false 0.0 on a labeler bench.

NEVER run torch and onnxruntime in one process: they fight over OpenMP and
whichever initialises first silently changes the other's numerics. Both imports
here are therefore lazy and backend-local -- keep them that way, and compare
backends by running this script twice.
"""
import argparse
import collections
import json
from pathlib import Path

MAP = {"GIVEN_NAME": "PER", "SURNAME": "PER", "COMPANY_NAME": "ORG",
       "CITY": "LOC", "STATE": "LOC", "COUNTRY": "LOC"}
NON_LATIN = ["ar", "zh", "ja", "ko", "ru", "hi", "el", "uk"]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return p * 100, r * 100, (2 * p * r / (p + r) * 100 if p + r else 0)


def coarse(l):
    l = l.lower()
    if (any(k in l for k in ("firstname", "lastname", "surname", "givenname", "name", "fullname"))
            and "user" not in l and "company" not in l and "domain" not in l):
        return "name"
    if any(k in l for k in ("company", "organization", "org")):
        return "org"
    if any(k in l for k in ("street", "address", "city", "state", "country", "secaddress", "building")):
        return "address"
    return "other"


class TorchModel:
    """HF checkpoint through torch. `tf32=False` forces true fp32 matmuls on
    Ampere, where torch otherwise silently rounds them to 10-bit mantissas."""

    def __init__(self, path, device, max_len=512, tokenizer=None, tf32=True):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        if not tf32:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.tok = AutoTokenizer.from_pretrained(tokenizer or path)
        # the saved tokenizer carries truncation from training (256); override it
        self.tok.model_max_length = max_len
        self.model = AutoModelForTokenClassification.from_pretrained(path).to(device).eval()
        self.id2label = self.model.config.id2label
        self.max_len = max_len
        self.device = device

    def predict(self, text):
        enc = self.tok(text, truncation=True, max_length=self.max_len,
                       return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.no_grad():
            pred = self.model(**enc).logits[0].argmax(-1).tolist()
        return pred, offs

    def close(self):
        del self.model
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()


class OnnxModel:
    """Exported model.onnx through onnxruntime. Deliberately avoids importing
    torch or transformers (OpenMP conflict, see module docstring) -- the raw
    `tokenizers` Tokenizer is what transformers wraps anyway, so token ids and
    offsets are identical to the torch path."""

    def __init__(self, path, device, max_len=512, tokenizer=None):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.np = np
        p = Path(path)
        onnx_file = p if p.is_file() else p / "model.onnx"
        tok_file = Path(tokenizer) if tokenizer else p / "tokenizer.json"
        if tok_file.is_dir():
            tok_file = tok_file / "tokenizer.json"

        self.tok = Tokenizer.from_file(str(tok_file))
        # mirrors the torch path's truncation=True, max_length=max_len
        self.tok.enable_truncation(max_length=max_len)
        self.tok.no_padding()

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_file), so,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        cfg = json.load(open(Path(onnx_file).parent / "config.json"))
        self.id2label = {int(k): v for k, v in cfg["id2label"].items()}
        self.max_len = max_len

    def predict(self, text):
        enc = self.tok.encode(text)
        ids = self.np.array([enc.ids], dtype=self.np.int64)
        feed = {"input_ids": ids}
        if "attention_mask" in self.inputs:
            feed["attention_mask"] = self.np.array([enc.attention_mask], dtype=self.np.int64)
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = self.np.zeros_like(ids)
        logits = self.sess.run(None, feed)[0]
        return logits[0].argmax(-1).tolist(), [list(o) for o in enc.offsets]

    def close(self):
        pass


class Model:
    """Backend-dispatching wrapper. Only `predict` differs between backends --
    span decoding below is shared, so a delta cannot come from the decoder."""

    def __init__(self, path, device, max_len=512, backend="torch",
                 tokenizer=None, tf32=True):
        if backend == "onnx":
            self.impl = OnnxModel(path, device, max_len, tokenizer)
        else:
            self.impl = TorchModel(path, device, max_len, tokenizer, tf32)
        self.id2label = self.impl.id2label

    def close(self):
        self.impl.close()

    def spans(self, text):
        pred, offs = self.impl.predict(text)

        out, cur = [], None

        def typ(l):
            return l[2:] if l[:2] in ("B-", "I-") else (None if l == "O" else l)

        def push():
            nonlocal cur
            if cur:
                t, s, e = cur
                # trim whitespace the tokenizer attributed to the token
                # (Metaspace gives the leading space to the piece -- this cost
                # us a bogus 41 F1 once)
                while s < e and text[s].isspace():
                    s += 1
                while e > s and text[e - 1].isspace():
                    e -= 1
                if e > s:
                    out.append([t, s, e])
                cur = None

        for i, p in enumerate(pred):
            if i >= len(offs):
                break
            a, b = offs[i]
            lab = self.id2label[int(p)]
            t = typ(lab)
            if a == b:
                continue
            if t is None:
                push()
                continue
            if cur and cur[0] == t:
                cur[2] = b
            else:
                push()
                cur = [t, a, b]
        push()
        return out


def span_f1(m, rows):
    """Exact (start,end) span match, label-agnostic."""
    tp = fp = fn = 0
    for r in rows:
        P = {(s, e) for _, s, e in m.spans(r["text"])}
        G = {(e["start"], e["end"]) for e in (r.get("entities") or [])}
        tp += len(P & G)
        fp += len(P - G)
        fn += len(G - P)
    return prf(tp, fp, fn)


def wikiann_char_f1(m, rows):
    by_lang = collections.defaultdict(list)
    for r in rows:
        by_lang[r["lang"]].append(r)
    res = {}
    for lang, rs in by_lang.items():
        ctp = cfp = cfn = 0
        for r in rs:
            sp = [[MAP[t], s, e] for t, s, e in m.spans(r["text"]) if t in MAP]
            sp.sort(key=lambda x: x[1])
            mg = []
            for t, s, e in sp:
                # merge adjacent same-type spans (GIVEN_NAME+SURNAME -> one PER)
                if mg and s - mg[-1][2] <= 1 and r["text"][mg[-1][2]:s].strip() == "":
                    mg[-1][2] = e
                else:
                    mg.append([t, s, e])
            pc, gc = set(), set()
            for _, s, e in mg:
                pc.update(range(s, e))
            for e_ in r["entities"]:
                gc.update(range(e_["start"], e_["end"]))
            ctp += len(pc & gc)
            cfp += len(pc - gc)
            cfn += len(gc - pc)
        res[lang] = prf(ctp, cfp, cfn)[2]
    return sum(res[l] for l in NON_LATIN if l in res) / max(len([l for l in NON_LATIN if l in res]), 1), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--indist-n", type=int, default=1000)
    ap.add_argument("--max-len", type=int, default=512,
                    help="inference truncation length. NOTE: the tokenizer.json saved by "
                         "training bakes in truncation at the training --max-length (256), "
                         "which silently caps inference; this overrides it explicitly.")
    ap.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer dir/file override. Needed for exported -onnx dirs, "
                         "which ship no tokenizer: point it at the source checkpoint. "
                         "A vocab-PRUNED model must use its own remapped tokenizer.")
    ap.add_argument("--no-tf32", action="store_true",
                    help="force true fp32 matmuls (torch/cuda defaults to TF32 on Ampere)")
    args = ap.parse_args()

    fresh = []
    for f in sorted(Path("data/testsets").glob("*.jsonl")):
        fresh += [json.loads(l) for l in open(f)]
    indist = [json.loads(l) for l in open("data/pii-bank.test.jsonl")][:args.indist_n]
    ai4 = [json.loads(l) for l in open("data/ai4privacy_val_sample.jsonl")]
    wa = [json.loads(l) for l in open("data/wikiann_ood.jsonl")]

    print(f"{'model':16s} | {'nameR':>5s} {'orgR':>5s} | {'P':>4s} {'R':>4s} {'F1':>5s} | "
          f"{'indist':>6s} {'ai4':>5s} {'nonLat':>6s}")
    print("-" * 79)
    for spec in args.models:
        name, path = spec.split("=", 1)
        m = Model(path, args.device, max_len=args.max_len, backend=args.backend,
                  tokenizer=args.tokenizer, tf32=not args.no_tf32)
        tp = fp = fn = 0
        byt = collections.defaultdict(lambda: [0, 0])
        for r in fresh:
            cov = set()
            for _, s, e in m.spans(r["text"]):
                cov.update(range(s, e))
            gc = set()
            for e in r["entities"]:
                sp = set(range(e["start"], e["end"]))
                gc |= sp
                c = coarse(e["label"])
                byt[c][0] += (len(sp & cov) / max(len(sp), 1) >= 0.5)
                byt[c][1] += 1
            tp += len(cov & gc)
            fp += len(cov - gc)
            fn += len(gc - cov)
        p, r_, f = prf(tp, fp, fn)
        nr = 100 * byt["name"][0] / max(byt["name"][1], 1)
        org = 100 * byt["org"][0] / max(byt["org"][1], 1)
        nl, _ = wikiann_char_f1(m, wa)
        print(f"{name:16s} | {nr:5.0f} {org:5.0f} | {p:4.0f} {r_:4.0f} {f:5.1f} | "
              f"{span_f1(m, indist)[2]:6.1f} {span_f1(m, ai4)[2]:5.1f} {nl:6.1f}", flush=True)
        m.close()
        del m


if __name__ == "__main__":
    main()
