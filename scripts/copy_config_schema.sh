#!/usr/bin/env bash
set -euo pipefail

# Get the repository name (directory name of the git root)
REPO_NAME=$(basename "$(git rev-parse --show-toplevel)")

# Use SCHEMAS_DIR env variable if available, otherwise fallback to $HOME/byteowlz/schemas
SCHEMAS_DIR="${SCHEMAS_DIR:-$HOME/byteowlz/schemas}"

# Construct destination directory with repo name
DEST_DIR="$SCHEMAS_DIR/$REPO_NAME"
DEST_SCHEMA="$DEST_DIR/$REPO_NAME.config.schema.json"

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_SCHEMA="$SRC_DIR/examples/config.schema.json"

if [ ! -f "$SRC_SCHEMA" ]; then
  echo "Source schema not found: $SRC_SCHEMA" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"
cp "$SRC_SCHEMA" "$DEST_SCHEMA"
echo "Copied schema to $DEST_SCHEMA"

cd "$DEST_DIR"

# Check if there are any changes to commit
if ! git diff --quiet; then
  git add .
  git commit -m "feat: updated $REPO_NAME schema" || {
    echo "Schema repo is already up to date"
    exit 0
  }
  git push || {
    echo "Schema repo is already up to date"
    exit 0
  }
  echo "Committed and pushed schema changes"
else
  echo "Schema repo is already up to date"
fi
