#!/usr/bin/env bash
# Runs inline in tmux/herdr or an ordinary shell; never creates/attaches sessions.
set -euo pipefail
umask 077

usage() {
    printf '%s\n' \
        'Usage: review-pi-terms.sh [OPTIONS]' \
        '' \
        'Discover Pi-session vocabulary and create a private offline review page.' \
        'No sessions are changed; no models, training, uploads or automatic approvals.' \
        '' \
        '  --sessions DIR       Input tree (default: ~/.pi/agent/sessions)' \
        '  --binary FILE        Recursive-capable NYM; otherwise NYM_BINARY or latest local build' \
        '  --output-dir DIR     New private run directory; must not already exist' \
        '  --no-open            Do not open the review page' \
        '  --limit N            Displayed suggestions (default: 200)' \
        '  --max-distinct N     Counting budget (default: 200000)' \
        '  --phrase-words N     Phrase size, 1..3 (default: 1 for lighter discovery)' \
        '  --min-count N        Minimum literal occurrences (default: 2)' \
        '  --exclude-file FILE  Exclude an exact file, repeatable; use for reserved holdouts' \
        '  -h, --help           Show help' \
        '' \
        'Counting/file/unit limits still apply; failures do not produce partial decks.'
}

fail() { printf 'error: %s\n' "$1" >&2; exit 1; }
supports_recursive() {
    local help
    [[ -x "$1" ]] || return 1
    help=$("$1" terms discover --help 2>/dev/null) || return 1
    [[ "$help" == *'--recursive'* ]]
}

sessions="${HOME:?HOME is required}/.pi/agent/sessions"
binary="${NYM_BINARY:-}"
run_dir=""
open_review=true
limit=200
max_distinct=200000
phrase_words=1
min_count=2
exclusions=()
while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --no-open) open_review=false; shift ;;
        --sessions|--binary|--output-dir|--limit|--max-distinct|--phrase-words|--min-count|--exclude-file)
            (($# >= 2)) || fail 'option requires a value'
            case "$1" in
                --sessions) sessions=$2 ;;
                --binary) binary=$2 ;;
                --output-dir) run_dir=$2 ;;
                --limit) limit=$2 ;;
                --max-distinct) max_distinct=$2 ;;
                --phrase-words) phrase_words=$2 ;;
                --min-count) min_count=$2 ;;
                --exclude-file) exclusions+=(--exclude-file "$2") ;;
            esac
            shift 2 ;;
        *) fail 'unknown option; use --help' ;;
    esac
done
[[ -d "$sessions" && ! -L "$sessions" ]] || fail 'sessions must be an existing real directory'

if [[ -n "$binary" ]]; then
    supports_recursive "$binary" || fail 'selected NYM binary does not support recursive discovery'
else
    build_root="${XDG_DATA_HOME:-$HOME/.local/share}/nym/builds"
    for candidate in "$build_root"/*/nym; do
        [[ -x "$candidate" ]] || continue
        if [[ -z "$binary" || "$candidate" -nt "$binary" ]]; then
            if supports_recursive "$candidate"; then binary=$candidate; fi
        fi
    done
    if [[ -z "$binary" ]]; then
        candidate=$(command -v nym || true)
        if [[ -n "$candidate" ]] && supports_recursive "$candidate"; then binary=$candidate; fi
    fi
    [[ -n "$binary" ]] || fail 'no recursive-capable NYM found; run just build-local default or pass --binary'
fi
# Keep relative executable paths usable after opening artifacts or changing shells.
if [[ "$binary" != /* ]]; then
    binary="$(cd "$(dirname "$binary")" && pwd -P)/$(basename "$binary")"
fi

if [[ -z "$run_dir" ]]; then
    run_dir="${XDG_STATE_HOME:-$HOME/.local/state}/nym/pi-terms/$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi
[[ ! -e "$run_dir" && ! -L "$run_dir" ]] || fail 'output run directory already exists; choose a new directory'
mkdir -p "$(dirname "$run_dir")"
mkdir -m 700 "$run_dir"
run_dir=$(cd "$run_dir" && pwd -P)
printf '[ner]\nenabled = false\n[decision]\nenabled = false\n' > "$run_dir/config.toml"

"$binary" --config "$run_dir/config.toml" terms discover "$sessions" \
    --recursive --extension jsonl,ndjson \
    --include '**.text' --include '**.thinking' \
    --phrase-words "$phrase_words" --max-distinct "$max_distinct" \
    --min-count "$min_count" --limit "$limit" \
    ${exclusions[@]+"${exclusions[@]}"} --output "$run_dir/discovery.json"
"$binary" --config "$run_dir/config.toml" terms review "$run_dir/discovery.json" \
    --html --output "$run_dir/review.html"
printf '\nReview: %s\nDiscovery: %s\n' "$run_dir/review.html" "$run_dir/discovery.json"
printf 'Read contexts, then download the review JSON. No terms are approved or installed automatically.\n'
if [[ "$open_review" == true ]]; then
    if command -v open >/dev/null 2>&1; then
        open "$run_dir/review.html" || printf 'Could not open a browser; open the review file manually.\n' >&2
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$run_dir/review.html" >/dev/null 2>&1 || printf 'Could not open a browser; open the review file manually.\n' >&2
    else
        printf 'No desktop opener found; open the review file manually.\n'
    fi
fi
