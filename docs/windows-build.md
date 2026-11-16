# Native Windows build

Since `gline-rs` became opt-in (the `ner-gliner` feature, trx-zp2t), the default
build no longer pulls in `esaxx-rs` at all and links natively on Windows. The
shim below is only needed when building with `--features ner-gliner`.

For `ner-gliner` builds: upstream `esaxx-rs` 0.1.10 hardcodes the static CRT
(`/MT`) while the prebuilt ONNX Runtime objects are dynamically linked (`/MD`),
which fails at link time (`LNK2038 RuntimeLibrary` mismatch, trx-rby8).
`esaxx-rs` reaches the build through the forked `gline-rs` -> `tokenizers`
default features; `esaxx` is only used by the tokenizers Unigram trainer, never
by nym inference.

Until the upstream fix lands, build with the local vendor shim (gitignored under the existing `/vendor/` convention, so it is never committed):

```bash
# one-time setup (Cargo registry source may live under a different hash directory)
mkdir -p vendor
cp -R ~/.cargo/registry/src/*/esaxx-rs-0.1.10 vendor/esaxx-rs
# in vendor/esaxx-rs/build.rs replace both ".static_crt(true)" with ".static_crt(false)"
cargo build --release --config 'patch.crates-io.esaxx-rs.path="vendor/esaxx-rs"'
```

`static_crt` only affects MSVC targets, so macOS and Linux builds are unchanged. Verify the lockfile diff afterwards: only the `esaxx-rs` entry changes (registry source and checksum replaced by the path).

Decision still open on trx-rby8: land an upstream `esaxx-rs`/`gline-rs` fix (preferred), or drop the tokenizers `http` feature in the `gline-rs` fork. `ner-load-dynamic` is not equivalent (it changes runtime distribution).

## Windows test status

With the shim, the full `cargo test --no-fail-fast` suite passes natively on Windows 11 (241 unit tests, 19 integration suites) as of the traces-cache lock fix, including the four real-ONNX cached-model tests (`NYM_TEST_CACHED_NER_MODEL=<path to model dir>`). The `terms runs` CLI reports verbatim `\\?\C:\...` canonical paths on Windows; that is cosmetic and tracked with trx-80g3 follow-ups if it matters to a consumer.
