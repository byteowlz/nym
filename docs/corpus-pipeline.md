# The corpus pipeline

The agent-facing recipe for this pipeline is the `nym-corpus` skill in
[byteowlz/skillissues](https://github.com/byteowlz/skillissues/tree/main/skills/nym-corpus).
Keep it there as the single copy.

One package, one ledger, gates at the end. nym owns collection, transformation and
accounting; publication decisions stay explicit, and nothing certifies anonymity.

```
sources (pi | claude | codex | opencode | cursor | directory tree)
   -> inventory        every unit, unresolved by default, deny rules evaluated
   -> propose          terms discover + optional model categorisation (advisory)
   -> review           approve / retain / deny per unit, tap-through page
   -> transform        fakes | stable placeholders | literals | assets
   -> verify           ledger, hashes, size accounting, external secret verifier
   -> export/gate      manifest for publication; deny and unresolved block
```

## What nym owns

| Stage | Command | Invariant |
|---|---|---|
| Inventory | `corpus inventory` | every file, key, value, opaque asset is a unit; nothing silently skipped |
| Deny gate | `corpus inventory --deny-file` | a deny match disqualifies the source; it is never redacted into acceptability |
| Review state | `corpus status` | counts only; original values stay private |
| Validate | `corpus validate` | sources unchanged by hash, decisions complete, deny clear |
| Transform | `corpus apply` | staged bundle, source-bound ledger, no partial output |
| Verify | `corpus verify` | every unit and byte reconciled against the ledger |

Borrowed and absorbed here: pi-share-hf's deny-regex gate (nothing else in the
pipeline can override it), pi-brain's stable placeholders (planned as a strategy),
and both projects' hash-keyed incremental state (already the ledger's basis).

## What nym deliberately does not absorb

- **Uploading.** Publishing to Hugging Face or an HTTP target is an external
  action with its own credentials and authority; nym emits a manifest instead.
- **A model as judge.** A decision head measured 0.35 recall against the
  operator's own labels, so models may propose and rank, never approve.
- **Rights decisions.** Ownership, licensing and client confidentiality are
  declared per source and reviewed by a human; nym records the declaration.

## Gates

1. Any source matching a deny rule blocks `validate`, `apply` and `verify`.
2. Any unresolved unit or missing review receipt blocks `validate` and `apply`.
3. Any source hash drift blocks everything after inventory.
4. `apply` writes a private ledger before the candidate bundle is committed; a
   failure leaves no bundle and no partial ledger.
5. The public manifest carries transformed-output hashes and counts only.

## Roadmap slices

1. **Source adapters** — accept pi / claude / codex / opencode / cursor session
   directories and normalise them into the inventory (pi is already parsed).
2. **Stable placeholders** — indexed per file (`<EMAIL_1>`) or bound to the value
   across the corpus (`<EMAIL_8f3a2b9c1d4e>`), for training data where realistic fakes
   would teach the wrong thing.
3. **External secret verifier** — `verify --secrets-cmd`, blocking on any
   finding, so a third-party scanner can veto a bundle nym believes is clean.
4. **Residual review** — a review page over a stratified sample of transformed
   output asking "is anything here unsafe to publish", recording verdicts.
5. **Export formats** — sessions, chatml, sft-jsonl, with the ledger bound in.
