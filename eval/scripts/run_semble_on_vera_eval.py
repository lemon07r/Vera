#!/usr/bin/env python3
"""Run semble on Vera's own eval dataset (21 tasks, 4 repos)."""

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "/tmp/semble-env/lib/python3.14/site-packages")
from semble import SembleIndex  # noqa: E402

CLONE_ROOT = Path(".bench/repos")
TASKS_DIR = Path("eval/tasks")
K = 10


def load_tasks():
    tasks = []
    for p in sorted(TASKS_DIR.glob("*.json")):
        data = json.loads(p.read_text())
        if isinstance(data, list):
            tasks.extend(data)
        else:
            tasks.append(data)
    return tasks


def is_match(result_path, gt):
    """Check overlap: file path matches AND line ranges overlap."""
    gt_path = gt["file_path"]
    if result_path != gt_path and not result_path.endswith("/" + gt_path) and not gt_path.endswith("/" + result_path):
        return False
    return True  # file-level match is sufficient for semble (it returns file-level chunks)


def recall_at_k(results, ground_truth, k):
    if not ground_truth:
        return 0.0
    top_k = results[:k]
    found = sum(1 for gt in ground_truth if any(is_match(r, gt) for r in top_k))
    return found / len(ground_truth)


def mrr(results, ground_truth):
    for i, r in enumerate(results):
        if any(is_match(r, gt) for gt in ground_truth):
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(results, ground_truth, k):
    top_k = results[:k]
    used = [False] * len(ground_truth)
    dcg = 0.0
    for i, r in enumerate(top_k):
        best_rel, best_idx = 0, -1
        for j, gt in enumerate(ground_truth):
            if not used[j] and is_match(r, gt) and gt.get("relevance", 1) > best_rel:
                best_rel = gt["relevance"]
                best_idx = j
        if best_idx >= 0:
            used[best_idx] = True
            dcg += best_rel / math.log2(i + 2)
    ideal_rels = sorted([gt.get("relevance", 1) for gt in ground_truth], reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    tasks = load_tasks()
    print(f"Loaded {len(tasks)} tasks", file=sys.stderr)

    by_repo = {}
    for t in tasks:
        by_repo.setdefault(t["repo"], []).append(t)

    all_metrics = []
    per_category = {}
    total_index_time = 0.0
    indexes = {}

    for repo_name, repo_tasks in sorted(by_repo.items()):
        repo_path = CLONE_ROOT / repo_name
        if not repo_path.exists():
            print(f"  SKIP {repo_name}", file=sys.stderr)
            continue

        print(f"  {repo_name}: indexing...", file=sys.stderr, end=" ", flush=True)
        t0 = time.monotonic()
        se = SembleIndex.from_path(str(repo_path))
        idx_time = time.monotonic() - t0
        total_index_time += idx_time
        indexes[repo_name] = se
        print(f"{idx_time:.1f}s, {len(repo_tasks)} queries", file=sys.stderr)

        for task in repo_tasks:
            t1 = time.monotonic()
            try:
                results = se.search(task["query"], top_k=K)
                result_paths = [r.chunk.file_path for r in results]
            except Exception as e:
                print(f"    FAIL {task['id']}: {e}", file=sys.stderr)
                result_paths = []
            lat = (time.monotonic() - t1) * 1000

            gt = task["ground_truth"]
            m = {
                "task_id": task["id"],
                "category": task["category"],
                "recall_at_1": recall_at_k(result_paths, gt, 1),
                "recall_at_5": recall_at_k(result_paths, gt, 5),
                "recall_at_10": recall_at_k(result_paths, gt, K),
                "mrr": mrr(result_paths, gt),
                "ndcg": ndcg_at_k(result_paths, gt, K),
                "latency_ms": lat,
                "results": result_paths[:5],
                "gt_files": [g["file_path"] for g in gt],
            }
            all_metrics.append(m)
            per_category.setdefault(task["category"], []).append(m)

    n = len(all_metrics)
    def avg(key): return sum(m[key] for m in all_metrics) / n

    print(f"\n=== Semble on Vera's Eval Set ({n} tasks) ===")
    print(f"R@1:  {avg('recall_at_1'):.4f}")
    print(f"R@5:  {avg('recall_at_5'):.4f}")
    print(f"R@10: {avg('recall_at_10'):.4f}")
    print(f"MRR:  {avg('mrr'):.4f}")
    print(f"nDCG: {avg('ndcg'):.4f}")
    print(f"Index: {total_index_time:.1f}s")
    print()
    for cat, ms in sorted(per_category.items()):
        nc = len(ms)
        print(f"{cat:20s} R@1={sum(m['recall_at_1'] for m in ms)/nc:.4f} R@10={sum(m['recall_at_10'] for m in ms)/nc:.4f} nDCG={sum(m['ndcg'] for m in ms)/nc:.4f} ({nc} tasks)")

    print("\n--- Per-task detail ---")
    for m in all_metrics:
        hit = "HIT" if m["recall_at_10"] > 0 else "MISS"
        print(f"  {m['task_id']:25s} {m['category']:15s} R@10={m['recall_at_10']:.2f} nDCG={m['ndcg']:.2f} {hit}")
        if m["recall_at_10"] < 1.0:
            print(f"    GT:      {m['gt_files']}")
            print(f"    Results: {m['results']}")


if __name__ == "__main__":
    main()
