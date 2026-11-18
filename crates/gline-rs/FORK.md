# gline-rs fork (2.0.0-rc.15 port)

Fork of `gline-rs` 1.1.0 (Apache-2.0, https://github.com/fbilhaut/gline-rs,
Fredérik Bilhaut), carried in-tree for the same reason as ../orp: the crates.io
release pins `ort` to `=2.0.0-rc.9`, which conflicts with nym's direct `ort`
dependency once the GLiNER engine and the token NER backend must share one
runtime (trx-zp2t).

Changes against 1.1.0:

- `ort`/`ort-sys` pins moved to `=2.0.0-rc.15`.
- `ndarray` dependency moved 0.16 -> 0.17 to match ort rc.15 (tensor inputs and
  extracted views must be the same ndarray major).
- Tensor inputs use explicit ORT values; output decoders use the rc.15
  extraction APIs. Outputs consume `orp::pipeline::OwnedOutputs`, allowing
  the native session lock to be released before post-processing.
- `publish = false`; consumed only as a path dependency. The environment
  comes from the patched `orp`.

Upstreamable: same as ../orp/FORK.md.
