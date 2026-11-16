# 🧩 ORP: a Lightweight Rust Framework for Building ONNX Runtime Pipelines with ORT

## 💬 Introduction

`orp` is a lightweight framework designed to simplify the creation and execution of **O**NNX **R**untime **P**ipelines in Rust. Built on top of the [`🦀 ort`](https://ort.pyke.io) runtime and the [`🔗 composable`](https://github.com/fbilhaut/composable) crate, it provides an simple way to handle data pre- and post-processing, chain multiple ONNX models together, while encouraging code reuse and clarity.


## 🔨 Sample Use-Cases

* [`🌿 gline-rs`](https://github.com/fbilhaut/gline-rs): inference engine for GLiNER models
* [`🧲 gte-rs`](https://github.com/fbilhaut/gte-rs): text embedding and re-ranking

## ⚡️ GPU/NPU Inferences

The execution providers available in `ort` can be leveraged to perform considerably faster inferences on GPU/NPU hardware.

The first step is to pass the appropriate execution providers in `RuntimeParameters`. For example:

```rust
let rtp = RuntimeParameters::default().with_execution_providers([
    CUDAExecutionProvider::default().build()
]);
```

The second step is to activate the appropriate features (see related section below), otherwise ir may **silently fall-back** to CPU. For example:

```console
$ cargo run --features=cuda ...
```

Please refer to `doc/ORT.md` for details about execution providers.


## 📦 Crate Features

This create mirrors the following `ort` features:

* To allow for dynamic loading of ONNX-runtime libraries: `load-dynamic`
* To allow for activation of execution providers: `cuda`, `tensorrt`, `directml`, `coreml`, `rocm`, `openvino`, `onednn`, `xnnpack`, `qnn`, `cann`, `nnapi`, `tvm`, `acl`, `armnn`, `migraphx`, `vitis`, and `rknpu`.

## ⚙️ Dependencies

* [`ort`](https://ort.pyke.io): the ONNX runtime wrapper
* [`composable`](https://github.com/fbilhaut/composable): this crate is used to actually define the pre- and post-processing pipelines by composition or elementary steps, and can in turn be used to combine mutliple pipelines.