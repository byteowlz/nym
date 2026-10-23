---
name: nym-corpus
description: Prepare agent sessions or document corpora for training or publication with nym - discover recurring sensitive terms, build reviewed redaction profiles, gate sources with deny rules, transform with a private ledger and hand a hash-bound manifest to a consumer such as Kompressor. Use when asked to anonymize, scrub, sanitize or publish many sessions, logs or documents, to build a dictionary of terms to redact, or to prepare a training corpus.
---

# Prepare a corpus with nym

nym transforms and accounts. It never certifies anonymity, decides rights or
publishes. Every stage below either proposes or verifies; a human approves.

Check the installed binary first: `nym --version`, `nym terms dict --help`,
`nym corpus --help`. Use `--json` for machine output everywhere.

## Ground rules

- Work on a copy of the sources. Plans, ledgers, discovery and review files contain
  original values: keep them outside the source tree and outside git, never upload them.
- Never auto-approve. A model, a dictionary or a missing finding is never evidence
  that something is safe. Excluded or unreviewed terms are unmeasured, not benign.
- Nothing downloads implicitly. Dictionaries come from `nym terms dict pull`, models
  from the configured cache. Ask before any network action.
- Report aggregates (counts, classes, rates), not values, unless the user asks to see them.

## 1. Discover recurring terms

```bash
nym terms dict pull en de            # once; pinned, hash-verified, CC-BY-SA lists
nym terms discover SESSIONS --recursive --bounded \
  --dict system,en,de --exclude-shapes \
  [--exclude-words glossary.txt] [--demote-words registry.txt] \
  --ner-rank --limit 2000 --output PRIVATE/discovery.json
```

- `--dict` and `--exclude-words` remove known words before ranking; matching is
  case-aware so names that are also words survive.
- `--demote-words` is for large public registries (package or domain names): rank
  last, never exclude - they contain people's names.
- `--ner-rank` puts model-recognised entities first. Add `--specific` to prefer
  terms concentrated in few sources over ones spread across everything.

## 2. Review terms into a profile

```bash
nym terms review PRIVATE/discovery.json --html --output PRIVATE/review.html
# user decides sensitive / contextual / dismiss / unsure in the browser
nym terms export PRIVATE/discovery.json --review PRIVATE/review.json --output PRIVATE/terms.txt
nym profiles import default --list sensitive PRIVATE/terms.txt
```

Approved terms are applied deterministically by `anon`, `detect` and `traces anon`.
Product names that models mistake for people (Claude Opus, GPT-4) are suppressed by
nym's built-in list; add your own harmless terms with
`nym profiles add default --list allow TERM...`.

## 3. Transform sessions

For agent sessions, detection-driven:

```bash
nym traces anon SESSIONS -o PRIVATE/out --keys PRIVATE/keys --plan PRIVATE/plan.json
# user reviews the plan (pi-nym or scripts/review_trace_anon.py), then:
nym traces anon SESSIONS -o PRIVATE/out --keys PRIVATE/keys --apply-plan PRIVATE/approved.json
```

For training data prefer stable placeholders over realistic fakes:
`nym anon FILE --strategy placeholder --placeholder-scope file|corpus`
(`file`: `<EMAIL_1>` per file; `corpus`: `<EMAIL_8f3a2b>` derived from the value,
identical across files and runs).

## 4. Account and gate with the corpus workflow

```bash
nym corpus inventory WORKING -o PRIVATE/plan.json --scope session --deny-file deny.txt
nym corpus status PRIVATE/plan.json --json
# review: every source needs a review receipt, every unit an approved action
nym corpus validate PRIVATE/plan.json --source WORKING --json
nym corpus apply PRIVATE/plan.json --source WORKING -o CANDIDATE --ledger PRIVATE/ledger.json
nym corpus verify --ledger PRIVATE/ledger.json --source WORKING --bundle CANDIDATE
```

- `deny.txt` holds regexes for projects, clients or topics that must never ship. A
  match disqualifies the whole source; it is never redacted into acceptability.
- A blocked run explains itself in `error.reason` (`--json`) or on stderr with `-v`.
- A source left byte-identical needs `publish_verbatim: true` in its review, because
  retaining everything publishes the original file.

## 5. Hand off

Give the consumer `CANDIDATE/` with its `manifest.json` (transformed-source paths,
SHA-256 and sizes only). Teacher labels must bind those transformed hashes, never
hashes of the private originals. Rights, segmentation and training gates belong to
the consumer; publishing is a separate, explicitly authorised action.

## Known limits

- Recall is gated on planted canaries; natural-PII recall and precision on a new
  corpus are unmeasured until someone reviews a sample. Run a small pilot first.
- `corpus` unit decisions are explicit per unit and are not yet pre-filled from
  detector findings, so on large session corpora use `traces anon` for the
  transformation and `corpus` for inventory, deny gating and accounting of the result.
- Discovery ranking cannot be validated with canaries that occur in few sessions:
  identities recurring across many sessions only show up in a real review.
