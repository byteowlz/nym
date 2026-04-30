# Issues

## Open

### [trx-j0p5.4] Implement OpenAI privacy-filter backend with Viterbi decoding (P1, feature)
## Goal

Add a `PrivacyFilterBackend` that implements `NerBackend` and runs the OpenAI privacy-filter model (and its Nemotron fine-tune) with BIOES-constrained Viterbi decoding in Rust.

## Background
...


### [trx-j0p5.3] Implement OpenMed PII token-classification backend (P1, feature)
## Goal

Add an `OpenMedBackend` that implements `NerBackend` and runs OpenMed PII models via ONNX Runtime in Rust.

## Background
...


### [trx-j0p5.2] Add NER backend trait and routing (P1, feature)
## Goal

Create a `NerBackend` trait in nym so multiple NER engines (GLiNER, OpenMed PII, privacy-filter) can be selected at runtime via CLI flag or config.

## Current State
...


### [trx-j0p5.1] Evaluate OpenMed PII models for ONNX export (P1, task)
## Goal

Determine which OpenMed PII token-classification models export cleanly to ONNX and benchmark them for nym's use case.

## Background
...


### [trx-j0p5] OpenMed + privacy-filter NER backends (P1, epic)
## Context

OpenMed (github.com/maziyarpanahi/openmed, Apache 2.0) is a medical NLP toolkit with 33+ PII token-classification models and an integration of OpenAI's `openai/privacy-filter` model. Both are directly usable by nym for PII detection beyond regex + GLiNER.

nym currently has two detection paths:
...


### [trx-j0p5.5] Port OpenMed smart entity merger for cross-backend dedup (P2, feature)
## Goal

Port OpenMed's regex-based semantic entity merger to Rust. This post-processes NER output from ANY backend to merge fragmented predictions into coherent entities.

## Problem
...


## Closed

- [trx-zpkq] Port OpenMed smart entity merger (closed 2026-04-30)
- [trx-swmv] Add Nemotron privacy-filter support (closed 2026-04-30)
- [trx-3dw5] Integrate OpenMed PII model as NER backend (closed 2026-04-30)
- [trx-3tsd] Add backend abstraction layer for NER engines (closed 2026-04-30)
- [trx-syqq] Evaluate OpenMed PII models for ONNX export (closed 2026-04-30)
