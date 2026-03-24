# Vera Benchmarks

This page summarizes the public benchmark results for Vera's retrieval pipeline.

## Scope

The current public benchmark snapshot covers 17 tasks across three open-source repositories:

- `ripgrep` for Rust
- `flask` for Python
- `fastify` for TypeScript

The tasks cover five workload categories:

- symbol lookup
- intent search
- cross-file discovery
- config lookup
- disambiguation

## Retrieval Quality

| Metric | ripgrep | cocoindex-code | vector-only | Vera hybrid |
|--------|---------|----------------|-------------|-------------|
| Recall@1 | 0.1548 | 0.1587 | 0.0952 | **0.4265** |
| Recall@5 | 0.2817 | 0.3730 | 0.4921 | **0.6961** |
| Recall@10 | 0.3651 | 0.5040 | 0.6627 | **0.7549** |
| MRR@10 | 0.2625 | 0.3517 | 0.2814 | **0.6009** |
| nDCG@10 | 0.2929 | 0.5206 | 0.7077 | **0.8008** |

Vera's strongest gains show up in top-of-list ranking quality. On this benchmark set, the hybrid pipeline improves MRR@10 by about 71% over the best non-Vera baseline and improves Recall@10 by about 14% over the strongest recall-oriented baseline.

## What The Numbers Mean

- BM25 helps with exact names, symbols, and disambiguation.
- Vector search helps with natural-language and intent-heavy queries.
- Reranking improves the order of the top candidates rather than just the recall of the candidate set.

That combination is why Vera tends to perform best on mixed workloads instead of excelling in only one query style.

## Performance Snapshot

From the same benchmark set:

| Measure | Result |
|---------|--------|
| BM25-only search p95 | `3.5 ms` |
| Hybrid search p95 | `6749 ms` |
| `ripgrep` index time | `65.1 s` |
| `flask` index time | `20.2 s` |
| `fastify` index time | `41.8 s` |

Two caveats matter here:

- BM25-only search is fast because it runs locally with no embedding or reranker round trips.
- API-backed hybrid search is slower because latency is dominated by remote model calls rather than local indexing or ranking work.

## Limits And Caveats

- The public benchmark summary uses the stable 17-task subset. A larger polyglot benchmark was not included in the public summary because API rate limits made it unreliable.
- The hybrid numbers above use remote embedding and reranking services. Local inference changes the latency profile.
- Benchmark data in this repository is intended to show comparative behavior, not guarantee exact performance on another machine or codebase.

## Related Docs

- Indexing performance note: [../benchmarks/indexing-performance.md](../benchmarks/indexing-performance.md)
- Reproduction guide: [../benchmarks/reports/reproduction-guide.md](../benchmarks/reports/reproduction-guide.md)
