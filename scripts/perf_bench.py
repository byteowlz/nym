#!/usr/bin/env python3
"""Measure nym detect latency / CPU / peak RSS across NER backends.

Cross-platform (Linux + macOS). Reports median wall-clock, CPU %, and peak
resident memory per run, using the child process's own rusage.

Usage:
    # regex-only baseline + one run per config file you pass:
    python3 scripts/perf_bench.py INPUT.txt [config1.toml config2.toml ...]

Environment:
    NYM_BIN        path to the nym binary   (default: ./target/release/nym)
    NYM_PERF_RUNS  repeats per config       (default: 3, reports median)

Build the binary first, e.g.:
    cargo build --release --features ner          # Linux/macOS, CPU
    cargo build --release --features ner-coreml   # macOS, Apple Neural Engine/GPU
    cargo build --release --features ner-cuda     # NVIDIA GPU
"""
import os, sys, time, statistics

BIN = os.environ.get("NYM_BIN", "./target/release/nym")
RUNS = int(os.environ.get("NYM_PERF_RUNS", "3"))

# ru_maxrss is in kilobytes on Linux, bytes on macOS/BSD.
_RSS_DIV = 1024.0 if sys.platform == "darwin" else 1.0  # -> KB
def rss_mb(ru_maxrss):
    return (ru_maxrss / _RSS_DIV) / 1024.0

def measure(args):
    t0 = time.monotonic()
    pid = os.posix_spawn(
        BIN, [BIN, *args], os.environ,
        file_actions=[
            (os.POSIX_SPAWN_OPEN, 1, os.devnull, os.O_WRONLY, 0),
            (os.POSIX_SPAWN_OPEN, 2, os.devnull, os.O_WRONLY, 0),
        ],
    )
    _, status, ru = os.wait4(pid, 0)
    wall = time.monotonic() - t0
    cpu = ru.ru_utime + ru.ru_stime
    return wall, (100.0 * cpu / wall if wall > 0 else 0), rss_mb(ru.ru_maxrss), status == 0

def bench(label, args):
    rows = [measure(args) for _ in range(RUNS)]
    wall = statistics.median(r[0] for r in rows)
    cpu = statistics.median(r[1] for r in rows)
    rss = max(r[2] for r in rows)
    ok = all(r[3] for r in rows)
    print(f"{label:28s} {wall:9.3f}s  {cpu:6.0f}%  {rss:9.1f} MB  {'' if ok else 'ERR'}")

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    inp = sys.argv[1]
    configs = sys.argv[2:]
    print(f"\n### input: {inp}  ({os.path.getsize(inp)} bytes, median of {RUNS} runs)")
    print(f"{'config':28s} {'wall':>9s}  {'cpu':>6s}  {'peakRSS':>11s}")
    print("-" * 62)
    bench("regex-only", ["detect", inp])
    for cfg in configs:
        label = os.path.splitext(os.path.basename(cfg))[0]
        bench(label, ["detect", inp, "--config", cfg])

if __name__ == "__main__":
    main()
