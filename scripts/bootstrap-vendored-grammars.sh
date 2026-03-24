#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

download_if_missing() {
    local target="$1"
    local url="$2"

    if [ ! -f "$target" ]; then
        mkdir -p "$(dirname "$target")"
        curl -fsSL "$url" -o "$target"
    fi
}

echo "[bootstrap] Ensuring vendored tree-sitter grammars are available"

download_if_missing \
    "crates/tree-sitter-vue/src/parser.c" \
    "https://raw.githubusercontent.com/tree-sitter-grammars/tree-sitter-vue/main/src/parser.c"
download_if_missing \
    "crates/tree-sitter-vue/src/scanner.c" \
    "https://raw.githubusercontent.com/tree-sitter-grammars/tree-sitter-vue/main/src/scanner.c"
download_if_missing \
    "crates/tree-sitter-vue/src/tag.h" \
    "https://raw.githubusercontent.com/tree-sitter-grammars/tree-sitter-vue/main/src/tag.h"
download_if_missing \
    "crates/tree-sitter-vue/src/tree_sitter/parser.h" \
    "https://raw.githubusercontent.com/tree-sitter-grammars/tree-sitter-vue/main/src/tree_sitter/parser.h"
download_if_missing \
    "crates/tree-sitter-vue/src/tree_sitter/alloc.h" \
    "https://raw.githubusercontent.com/tree-sitter-grammars/tree-sitter-vue/main/src/tree_sitter/alloc.h"
download_if_missing \
    "crates/tree-sitter-vue/src/tree_sitter/array.h" \
    "https://raw.githubusercontent.com/tree-sitter-grammars/tree-sitter-vue/main/src/tree_sitter/array.h"

download_if_missing \
    "crates/tree-sitter-dockerfile/src/parser.c" \
    "https://raw.githubusercontent.com/camdencheek/tree-sitter-dockerfile/main/src/parser.c"
download_if_missing \
    "crates/tree-sitter-dockerfile/src/scanner.c" \
    "https://raw.githubusercontent.com/camdencheek/tree-sitter-dockerfile/main/src/scanner.c"
download_if_missing \
    "crates/tree-sitter-dockerfile/src/tree_sitter/parser.h" \
    "https://raw.githubusercontent.com/camdencheek/tree-sitter-dockerfile/main/src/tree_sitter/parser.h"

download_if_missing \
    "crates/tree-sitter-astro/src/parser.c" \
    "https://raw.githubusercontent.com/virchau13/tree-sitter-astro/master/src/parser.c"
download_if_missing \
    "crates/tree-sitter-astro/src/scanner.c" \
    "https://raw.githubusercontent.com/virchau13/tree-sitter-astro/master/src/scanner.c"
download_if_missing \
    "crates/tree-sitter-astro/src/tag.h" \
    "https://raw.githubusercontent.com/virchau13/tree-sitter-astro/master/src/tag.h"
download_if_missing \
    "crates/tree-sitter-astro/src/tree_sitter/parser.h" \
    "https://raw.githubusercontent.com/virchau13/tree-sitter-astro/master/src/tree_sitter/parser.h"
download_if_missing \
    "crates/tree-sitter-astro/src/tree_sitter/alloc.h" \
    "https://raw.githubusercontent.com/virchau13/tree-sitter-astro/master/src/tree_sitter/alloc.h"
download_if_missing \
    "crates/tree-sitter-astro/src/tree_sitter/array.h" \
    "https://raw.githubusercontent.com/virchau13/tree-sitter-astro/master/src/tree_sitter/array.h"

echo "[bootstrap] Vendored grammars ready"
