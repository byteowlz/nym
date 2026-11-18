# orp fork (2.0.0-rc.15 port)

Fork of `orp` 1.0.0 (Apache-2.0, https://github.com/fbilhaut/orp, Fredérik Bilhaut),
carried in-tree because the crates.io release pins `ort` to `=2.0.0-rc.9`, which
predates the CoreML MLProgram/static-shape/compute-unit provider options that nym's
token backend needs (trx-zp2t). `gline-rs` (see ../gline-rs) depends on `orp ^1`, so
this fork satisfies both without pulling the pinned rc.9.

Changes against 1.0.0:

- `ort`/`ort-sys` pins moved to `=2.0.0-rc.15` (rc.15 API: one explicit
  environment per process, `Session::builder(&env)`, EPs under `ort::ep`,
  `Session::inputs()`/`outputs()` accessors, `Outlet::name()`/`dtype()`).
- New `env` module: create-once process-global ONNX Runtime environment
  (`orp::env::global_environment`), because rc.15 allows exactly one
  environment and exposes no public accessor for an existing one. nym's token
  backend shares this environment in `ner-gliner` builds.
- Session access is synchronized. `Run` results become an owned map of ORT
  value handles before releasing the session lock, so post-processing cannot
  retain a borrowed mutable session. A synthetic native concurrency test covers
  simultaneous requests and results surviving subsequent runs.
- `publish = false`; consumed only as a path dependency.

Upstreamable: the environment module and rc.15 port could be proposed upstream;
the fork exists so nym does not depend on upstream release timing.
