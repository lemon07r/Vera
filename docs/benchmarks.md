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

### Local vs API Models

The local Jina models (239M embedding + 278M reranker, ONNX) are competitive with the much larger Qwen3-Embedding-8B API model on the same 17-task benchmark:

| Metric | Jina local (ONNX) | Qwen3-8B (API) |
|--------|-------------------|----------------|
| MRR@10 | **0.68** | 0.60 |
| Recall@5 | 0.65 | **0.73** |
| Recall@10 | 0.73 | **0.75** |
| nDCG@10 | 0.72 | **0.81** |

Jina local ranks the best result higher (better MRR), while the API model retrieves more relevant results overall (better recall and nDCG). For most use cases, the local models are accurate enough to skip the API entirely.

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

## Vera vs ColGREP (Late Interaction)

21 tasks across 4 repos (ripgrep, flask, fastify, turborepo). Vera used Jina v5 nano + Jina reranker on CUDA (RTX 4080). ColGREP used LateOn-Code models on CPU. Qwen3-8B API results are from the 17-task/3-repo benchmark (different task set, included for reference).

| Metric | Vera (Jina CUDA) | ColGREP (149M) | ColGREP Edge (17M) | Qwen3-8B API* |
|--------|------------------|----------------|--------------------|--------------:|
| Recall@1 | 0.476 | **0.571** | 0.524 | 0.43 |
| Recall@5 | **0.667** | **0.667** | 0.571 | 0.73 |
| Recall@10 | 0.667 | **0.714** | **0.714** | 0.75 |
| MRR@10 | 0.556 | **0.617** | 0.566 | 0.60 |
| nDCG@10 | 0.509 | **0.561** | 0.524 | 0.81 |

\* Qwen3-8B numbers are from a different task set (17 tasks, 3 repos) and are not directly comparable to the 21-task ColGREP columns. They are included to show how the API model performs on a similar workload.

Indexing time (4 repos combined, release builds):

| Tool | Total time | Hardware |
|------|-----------|----------|
| Vera (Jina CUDA) | ~79s | RTX 4080 |
| ColGREP (149M, CPU) | ~180s | Ryzen 5 7600X3D 6c/12t |
| ColGREP Edge (17M, CPU) | ~160s | Ryzen 5 7600X3D 6c/12t |

ColGREP's late-interaction (ColBERT) approach scores higher on Recall@1 and MRR despite using smaller models and no GPU. Vera's strengths are broader language support (63 vs 18), agent integration (skill files, MCP server), and hybrid BM25+vector fusion. ColGREP indexes are 30-40% larger due to multi-vector storage.

## Model Evaluation Notes

We evaluated alternative models to see if better options exist for local inference.

### Reranker: Jina v2 vs GTE-ModernBERT-base (int8)

Both tests used Jina v5 nano CUDA embeddings. GTE re-reranked Jina's top-30 candidates.

| Metric | Jina reranker (278M) | GTE-ModernBERT (149M) |
|--------|---------------------|----------------------|
| Recall@1 | **0.42** | 0.19 |
| Recall@5 | 0.69 | **0.72** |
| MRR@10 | **0.70** | 0.51 |
| nDCG@10 | **0.68** | 0.59 |

Jina reranker wins on the metrics that matter most for code search (MRR, Recall@1. putting the right result first). GTE's slight Recall@5 edge is noise.

CPU reranking speed (Ryzen 5 7600X3D, 6c/12t):

| Reranker | 10 docs | Per doc |
|----------|---------|--------|
| Jina v2 (278M, int8) | **1.0s** | **104ms** |
| GTE-ModernBERT (149M, int8) | 1.5s | 145ms |

Jina is faster despite being larger. likely due to its simpler architecture vs ModernBERT's Flash Attention overhead on CPU.

### Embedding: Why Jina v5 nano stays

Jina-embeddings-v5-text-nano (239M, Feb 2026) scores 71.0 on MTEB English v2. the highest among models under 500M parameters. Alternatives considered:

| Model | Params | MTEB English | Code-specific? | Notes |
|-------|--------|-------------|----------------|-------|
| **Jina v5 nano** | **239M** | **71.0** | No (general) | Current default |
| EmbeddingGemma-300M | 308M | 68.4 | No | Larger, lower score |
| CodeRankEmbed | 137M | N/A | Yes (code-only) | Trained on code pairs, not NL→code queries |
| snowflake-arctic-embed-xs | 22M | ~42 (est.) | No | 10x smaller but much lower quality |

CodeRankEmbed (137M) is a bi-encoder that could be a drop-in replacement, but it's trained on code-to-code matching (CoRNStack dataset), not natural-language-to-code retrieval. Vera's primary use case is queries like "how does the search pipeline work". general models handle this better than code-specialized ones.

### CPU indexing speed in context

~6 minutes for ~3,100 chunks on a 6-core CPU is expected for a 239M-parameter model. The bottleneck is pure matrix math in the embedding model, not file I/O or parsing (which takes <2 seconds). Indexing time scales linearly with chunk count:

- ~500-line project: ~30s on CPU
- ~23K-line project (Vera itself): ~6 min on CPU, ~8s on CUDA
- After initial index, `vera update .` only re-embeds changed files

These models are designed for GPU inference. CPU mode works but is not their intended target. For large initial indexes, use `--onnx-jina-cuda` or API mode.

## Limits And Caveats

- The public benchmark summary uses the stable 17-task subset. A larger polyglot benchmark was not included in the public summary because API rate limits made it unreliable.
- The hybrid numbers above use remote embedding and reranking services. Local CUDA inference indexes the same repos in seconds (e.g., ~8s for ripgrep on an RTX 4080); CPU-only takes ~6 min on a Ryzen 5 7600X3D (6c/12t).
- Benchmark data in this repository is intended to show comparative behavior, not guarantee exact performance on another machine or codebase.

## Related Docs

- Indexing performance note: [../benchmarks/indexing-performance.md](../benchmarks/indexing-performance.md)
- Reproduction guide: [../benchmarks/reports/reproduction-guide.md](../benchmarks/reports/reproduction-guide.md)
