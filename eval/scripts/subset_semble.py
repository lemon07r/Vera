#!/usr/bin/env python3
"""Create a representative subset of semble tasks for faster benchmarking.

Picks repos that cover diverse languages while keeping the task count manageable.
"""

import json
import sys
from pathlib import Path

# Representative subset: 2-3 repos per language family, covering the most popular ones
SUBSET_REPOS = [
    # Python
    "fastapi", "flask", "requests",
    # JavaScript/TypeScript
    "express", "zod", "axios",
    # Rust
    "tokio", "serde", "axum",
    # Go
    "gin", "cobra",
    # Java
    "gson",
    # C/C++
    "curl", "fmtlib",
    # Ruby
    "sinatra",
    # Elixir
    "phoenix",
]

def main():
    tasks_dir = Path("eval/tasks/semble")
    out_dir = Path("eval/tasks/semble-subset")
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for repo in SUBSET_REPOS:
        src = tasks_dir / f"{repo}.json"
        if not src.exists():
            print(f"  SKIP {repo} (no annotation file)", file=sys.stderr)
            continue
        tasks = json.loads(src.read_text())
        (out_dir / f"{repo}.json").write_text(json.dumps(tasks, indent=2) + "\n")
        total += len(tasks)
        print(f"  {repo}: {len(tasks)} tasks", file=sys.stderr)

    print(f"\nSubset: {total} tasks across {len(SUBSET_REPOS)} repos", file=sys.stderr)

    # Generate subset corpus.toml
    import tomllib
    with open("eval/semble-corpus.toml", "rb") as f:
        corpus = tomllib.load(f)

    subset_repos = [r for r in corpus["repos"] if r["name"] in SUBSET_REPOS]
    lines = [
        '# Semble benchmark subset (auto-generated)',
        '',
        '[corpus]',
        'version = 1',
        'description = "Semble benchmark subset for faster evaluation"',
        'clone_root = ".bench/semble-repos"',
        '',
    ]
    for repo in subset_repos:
        lines.append('[[repos]]')
        lines.append(f'name = "{repo["name"]}"')
        lines.append(f'url = "{repo["url"]}"')
        lines.append(f'commit = "{repo["commit"]}"')
        lines.append(f'language = "{repo["language"]}"')
        lines.append(f'description = "semble benchmark repo"')
        if repo.get("benchmark_root"):
            lines.append(f'benchmark_root = "{repo["benchmark_root"]}"')
        lines.append('')

    Path("eval/semble-subset-corpus.toml").write_text('\n'.join(lines))
    print(f"Corpus manifest: eval/semble-subset-corpus.toml", file=sys.stderr)


if __name__ == "__main__":
    main()
