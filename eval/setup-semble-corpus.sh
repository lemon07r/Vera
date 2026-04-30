#!/usr/bin/env bash
# Clone semble benchmark repos at pinned revisions.
# Idempotent: skips repos already at the correct SHA.
#
# Usage: ./eval/setup-semble-corpus.sh [--force]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CORPUS_FILE="$SCRIPT_DIR/semble-corpus.toml"

FORCE=false
[[ "${1:-}" == "--force" ]] && FORCE=true

if [[ ! -f "$CORPUS_FILE" ]]; then
    echo "ERROR: semble-corpus.toml not found. Run: python3 eval/scripts/convert_semble.py"
    exit 1
fi

CLONE_ROOT=$(python3 -c "
import tomllib, os
with open('$CORPUS_FILE', 'rb') as f:
    data = tomllib.load(f)
root = data['corpus']['clone_root']
if not os.path.isabs(root):
    root = os.path.join('$REPO_ROOT', root)
print(root)
")

echo "=== Semble Corpus Setup ==="
echo "Clone root: $CLONE_ROOT"
echo ""

mkdir -p "$CLONE_ROOT"

TOTAL=0
SKIPPED=0
CLONED=0

python3 -c "
import tomllib, json
with open('$CORPUS_FILE', 'rb') as f:
    data = tomllib.load(f)
for repo in data['repos']:
    print(json.dumps(repo))
" | while IFS= read -r repo_json; do
    NAME=$(echo "$repo_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['name'])")
    URL=$(echo "$repo_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['url'])")
    COMMIT=$(echo "$repo_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['commit'])")

    REPO_DIR="$CLONE_ROOT/$NAME"

    if [[ -d "$REPO_DIR/.git" ]] && [[ "$FORCE" != "true" ]]; then
        CURRENT_SHA=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo "")
        if [[ "$CURRENT_SHA" == "$COMMIT" ]]; then
            continue
        fi
        rm -rf "$REPO_DIR"
    elif [[ -d "$REPO_DIR" ]] && [[ "$FORCE" == "true" ]]; then
        rm -rf "$REPO_DIR"
    fi

    echo "Cloning $NAME..."
    git clone --quiet --depth 1 "$URL" "$REPO_DIR" 2>/dev/null || \
        git clone --quiet "$URL" "$REPO_DIR"
    git -C "$REPO_DIR" checkout --quiet "$COMMIT" 2>/dev/null || true

    ACTUAL=$(git -C "$REPO_DIR" rev-parse HEAD)
    if [[ "$ACTUAL" != "$COMMIT" ]]; then
        echo "  WARNING: $NAME at $ACTUAL (wanted $COMMIT), shallow clone may not have it"
        rm -rf "$REPO_DIR"
        git clone --quiet "$URL" "$REPO_DIR"
        git -C "$REPO_DIR" checkout --quiet "$COMMIT"
    fi
done

echo ""
echo "=== Semble corpus setup complete ==="
echo "Repos in $CLONE_ROOT:"
ls -1 "$CLONE_ROOT" | wc -l
