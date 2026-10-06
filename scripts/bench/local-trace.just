# Isolated offline recipes; no edits to the existing gate, fixtures or top justfile.
# Raw input, bundles, results and review exports MUST live outside ANY git tree.
set working-directory := "../.."
set positional-arguments

# Only the new owned Python suites; no Rust build/model/network required.
test:
    uv run --offline --no-project -m unittest discover -s scripts/datagen -p 'test_agent_trace_corpus.py' -v
    uv run --offline --no-project -m unittest discover -s scripts/bench -p 'test_local_trace_eval.py' -v

# Existing 81-label checkpoint config is required, not a guessed new label head.
corpus *args:
    uv run --offline --no-project scripts/datagen/agent_trace_corpus.py "$@"

# Explicit local JSONL paths/selectors/insertion plan; private 0700/0600 exports.
prepare *args:
    uv run --offline --no-project scripts/bench/local_trace_eval.py prepare "$@"

# Normalized detector/placeholder results supplied by the local caller.
score *args:
    uv run --offline --no-project scripts/bench/local_trace_eval.py score "$@"

# Excludes injected canaries from real-distribution precision review.
sample *args:
    uv run --offline --no-project scripts/bench/local_trace_eval.py sample "$@"

# Completed human reviews -> value/path/hash-free weighted category summary.
summarize *args:
    uv run --offline --no-project scripts/bench/local_trace_eval.py summarize "$@"
