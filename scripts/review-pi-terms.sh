#!/usr/bin/env bash
# Runs inline in tmux/herdr or an ordinary shell; never creates/attaches sessions.
set -euo pipefail
umask 077

usage() {
    printf '%s\n' \
        'Usage: review-pi-terms.sh [OPTIONS]' \
        '' \
        'Discover Pi-session vocabulary and create a private offline review page.' \
        'No sessions are changed; no new models, training, uploads or automatic approvals.' \
        '' \
        '  --sessions DIR       Input tree (default: ~/.pi/agent/sessions)' \
        '  --binary FILE        Recursive-capable NYM; otherwise NYM_BINARY or latest local build' \
        '  --output-dir DIR     New private run directory; must not already exist' \
        '  --no-open            Do not open the review page' \
        '  --serve              Share the new review on tailnet; foreground, Ctrl-C stops' \
        '  --serve-existing DIR Share DIR/review.html without rescanning or rewriting it' \
        '  --port N             Review server port (default: 8521; requires serving)' \
        '  --no-progress        Disable the stderr progress bar/stage counts' \
        '  --strict-unicode     Reject unpaired UTF-16 instead of explicit discovery projection' \
        '  --frequency-ranking Use raw spread ranking instead of source-specific suggestions' \
        '  --no-ner-rank        Disable the default local NER-positive review boost' \
        '  --ner-rank-model DIR Use an existing local/cached token model for the boost'  \
        '  --limit N            Displayed suggestions (default: 200)' \
        '  --max-distinct N     Bounded proposal slots (default: 200000)' \
        '  --phrase-words N     Phrase size, 1..3 (default: 1 for lighter discovery)' \
        '  --min-count N        Minimum literal occurrences (default: 2)' \
        '  --exclude-file FILE  Exclude an exact file, repeatable; use for reserved holdouts' \
        '  -h, --help           Show help' \
        '' \
        'Uses bounded proposals and exact recount: 100x display limit (10x with frequency ranking).' \
        'Local CPU NER at 0.5 boosts recognized candidates before the display cutoff.' \
        'No match is not benign; model matches still need human context review. Missing models fail.' \
        'Serving uses uv/Python stdlib and Tailscale, only when explicitly requested.' \
        'Completed runs are registered in XDG state; use nym terms runs list/show/open/serve.' \
        'Only a private review snapshot is shared, behind a capability link; no public funnel.' \
        'Proposals may omit rare literals. Omission is not evidence of safety.' \
        'File/unit limits still apply; failures do not produce partial decks.'
}

fail() { printf 'error: %s\n' "$1" >&2; exit 1; }
serve_page() {
    command -v uv >/dev/null 2>&1 || fail 'serving requires uv; offline review files remain available'
    local script_dir open_flags=()
    script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
    if [[ "$open_review" == true ]]; then open_flags=(--open); fi
    exec uv run --no-project python "$script_dir/serve_terms_review.py" "$1/review.html" \
        --tailnet --port "$port" ${open_flags[@]+"${open_flags[@]}"}
}

supports_recursive() {
    local help
    [[ -x "$1" ]] || return 1
    help=$("$1" terms discover --help 2>/dev/null) || return 1
    [[ "$help" == *'--recursive'* && "$help" == *'--bounded'* && "$help" == *'--progress'* && "$help" == *'--repair-unicode'* && "$help" == *'--specific'* && "$help" == *'--ner-rank'* ]] || return 1
    help=$("$1" terms runs --help 2>/dev/null) || return 1
    [[ "$help" == *'register'* && "$help" == *'serve'* ]]
}

sessions="${HOME:?HOME is required}/.pi/agent/sessions"
binary="${NYM_BINARY:-}"
run_dir=""
open_review=true
serve_review=false
existing_review=""
port=8521
port_explicit=false
discovery_options=false
progress_flag=--progress
unicode_flags=(--repair-unicode)
ranking_flags=(--specific)
ner_flags=(--ner-rank)
ner_model_flags=()
limit=200
max_distinct=200000
phrase_words=1
min_count=2
exclusions=()
while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --no-open) open_review=false; shift ;;
        --serve) serve_review=true; shift ;;
        --serve-existing|--port)
            (($# >= 2)) || fail 'option requires a value'
            if [[ "$1" == --port ]]; then port=$2; port_explicit=true; else existing_review=$2; fi
            shift 2 ;;
        --no-progress) progress_flag=--no-progress; discovery_options=true; shift ;;
        --strict-unicode) unicode_flags=(); discovery_options=true; shift ;;
        --frequency-ranking) ranking_flags=(); discovery_options=true; shift ;;
        --no-ner-rank) ner_flags=(); discovery_options=true; shift ;;
        --sessions|--binary|--output-dir|--limit|--max-distinct|--phrase-words|--min-count|--exclude-file|--ner-rank-model)
            (($# >= 2)) || fail 'option requires a value'
            if [[ "$1" != --binary ]]; then discovery_options=true; fi
            case "$1" in
                --sessions) sessions=$2 ;;
                --binary) binary=$2 ;;
                --output-dir) run_dir=$2 ;;
                --limit) limit=$2 ;;
                --max-distinct) max_distinct=$2 ;;
                --phrase-words) phrase_words=$2 ;;
                --min-count) min_count=$2 ;;
                --exclude-file) exclusions+=(--exclude-file "$2") ;;
                --ner-rank-model) ner_model_flags=(--ner-rank-model "$2") ;;
            esac
            shift 2 ;;
        *) fail 'unknown option; use --help' ;;
    esac
done
[[ -z "${ner_model_flags[*]-}" || -n "${ner_flags[*]-}" ]] || fail '--ner-rank-model cannot be used with --no-ner-rank'
[[ "$port" =~ ^[0-9]{1,5}$ ]] && ((10#$port <= 65535)) || fail 'invalid review server port'
if [[ -n "$existing_review" ]]; then
    [[ "$discovery_options" == false && "$serve_review" == false ]] || fail '--serve-existing cannot also request discovery'
    [[ -d "$existing_review" && ! -L "$existing_review" && -f "$existing_review/review.html" && ! -L "$existing_review/review.html" ]] || fail 'existing review must be a real directory with review.html'
    serve_page "$(cd "$existing_review" && pwd -P)"
fi
[[ "$port_explicit" == false || "$serve_review" == true ]] || fail '--port requires --serve or --serve-existing'
if [[ "$serve_review" == true ]]; then command -v uv >/dev/null 2>&1 || fail 'serving requires uv'; fi
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
    --recursive --bounded "$progress_flag" ${unicode_flags[@]+"${unicode_flags[@]}"} ${ranking_flags[@]+"${ranking_flags[@]}"} ${ner_flags[@]+"${ner_flags[@]}"} ${ner_model_flags[@]+"${ner_model_flags[@]}"} --extension jsonl,ndjson \
    --include '**.text' --include '**.thinking' \
    --phrase-words "$phrase_words" --max-distinct "$max_distinct" \
    --min-count "$min_count" --limit "$limit" \
    ${exclusions[@]+"${exclusions[@]}"} --output "$run_dir/discovery.json"
"$binary" --config "$run_dir/config.toml" terms review "$run_dir/discovery.json" \
    --html --output "$run_dir/review.html"
"$binary" --config "$run_dir/config.toml" terms runs register "$run_dir"
printf '\nReview: %s\nDiscovery: %s\n' "$run_dir/review.html" "$run_dir/discovery.json"
printf 'Read contexts, then download the review JSON. No terms are approved or installed automatically.\n'
if [[ "$serve_review" == true ]]; then serve_page "$run_dir"; fi
if [[ "$open_review" == true ]]; then
    if command -v open >/dev/null 2>&1; then
        open "$run_dir/review.html" || printf 'Could not open a browser; open the review file manually.\n' >&2
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$run_dir/review.html" >/dev/null 2>&1 || printf 'Could not open a browser; open the review file manually.\n' >&2
    else
        printf 'No desktop opener found; open the review file manually.\n'
    fi
fi
