# Vera — Final Recommendation Memo

**Date:** 2026-03-20
**Version:** 0.1.0

## 1. Chosen Architecture & Rationale

Vera uses a **hybrid BM25 + vector + RRF fusion + cross-encoder reranking** pipeline, implemented in Rust with SQLite-based storage. Every major decision was driven by benchmarks on Vera's own 21-task evaluation suite across 4 repositories.

### Technology Stack

| Component | Choice | Key Evidence |
|-----------|--------|-------------|
| **Language** | Rust 1.94 | 1.6–1.8× faster tree-sitter parsing than TypeScript/Bun, 10× faster cold start (0.5ms vs 5ms), single binary distribution (~15 MB). See [ADR-001](adr/001-implementation-language.md). |
| **Storage** | SQLite + sqlite-vec (vectors/metadata) + Tantivy (BM25) | 10ms vector query p50, 7.6K chunks/sec writes — both within budget. 60 crates vs 537 (LanceDB), 40s vs 150s build time. See [ADR-002](adr/002-storage-backend.md). |
| **Embedding** | Qwen3-Embedding-8B (4096→1024-dim via Matryoshka truncation) | Highest Recall@10 (0.66) and nDCG (0.71) on Vera's task suite, outperforming bge-en-icl and Qwen3-0.6B. See [ADR-003](adr/003-embedding-model.md). |
| **Chunking** | Symbol-aware (tree-sitter AST) with sliding-window fallback | 2.3× higher MRR on symbol lookup (0.55 vs 0.24), 14% more token-efficient than sliding-window. See [ADR-004](adr/004-chunking-strategy.md). |
| **Retrieval** | BM25 + vector + RRF fusion + Qwen3-Reranker | Hybrid achieves +111% MRR over BM25-only and +111% over vector-only. Reranking adds +77% MRR. No competitor combines all four stages. |

### Pipeline Architecture

```
Source Files → Tree-sitter Parser → Symbol-Aware Chunker → Embedding API
                                                              ↓
Query → [BM25 Search] + [Vector Search] → RRF Fusion → Reranker → Ranked Results
         (Tantivy)      (sqlite-vec)                    (API)
```

### Why This Architecture Wins

Vera's full hybrid pipeline achieves the best retrieval quality of any tool benchmarked:

| Metric | ripgrep | cocoindex-code | vector-only | **Vera** |
|--------|---------|----------------|-------------|----------|
| Recall@5 | 0.35 | 0.37 | 0.49 | **0.73** |
| MRR@10 | 0.32 | 0.35 | 0.28 | **0.59** |
| nDCG@10 | 0.29 | 0.52 | 0.71 | **0.80** |

Each component addresses a distinct failure mode:
- **BM25** rescues exact identifier lookup (symbol MRR: 0.75 BM25-only vs 0.24 vector-only)
- **Vector search** enables semantic queries where BM25 fails (config lookup: 0.00 BM25 vs 1.00 hybrid)
- **Reranking** converts high recall into high precision (+79% Precision@3)

## 2. Rejected Paths & Reasons

### TypeScript/Bun Implementation
**Rejected because:** 10× slower cold start (5ms vs 0.5ms), larger distribution (60–80 MB compiled binary vs ~15 MB), weaker ecosystem for core dependencies (no `ignore` crate equivalent for gitignore-aware walking, no Tantivy for BM25). See [ADR-001](adr/001-implementation-language.md).

**Trade-off accepted:** Slower development iteration speed, steeper learning curve for contributors.

### LanceDB Storage
**Rejected because:** Despite being 5× faster on vector queries (2ms vs 10ms) and 32× faster on writes, SQLite's performance is fully adequate for Vera's scale (<100K chunks). LanceDB's 537-crate dependency tree vs SQLite's 60 crates would have tripled build times, increased binary size, and required async everywhere. See [ADR-002](adr/002-storage-backend.md).

**Trade-off accepted:** Slower raw vector operations. Mitigated by query embedding cache and adequate performance budget.

### bge-en-icl Embedding Model
**Rejected because:** General-purpose model collapsed on symbol lookup (MRR 0.054) and config tasks (Recall@5 0.25) despite competitive intent search (Recall@5 0.70). Overall Recall@10 (0.33) was half of Qwen3-8B (0.66). See [ADR-003](adr/003-embedding-model.md).

### File-Level and Sliding-Window Chunking as Primary Strategy
**Rejected because:** File-level chunks dilute relevance signal (MRR 0.26 vs symbol-aware 0.38). Sliding-window splits code at arbitrary boundaries (symbol lookup MRR 0.24 vs 0.55). Both waste tokens by including irrelevant surrounding code. See [ADR-004](adr/004-chunking-strategy.md).

**Sliding-window retained as Tier 0 fallback** for files without tree-sitter grammar support.

### Graph-Heavy Architecture (à la grepai RPG)
**Deferred, not rejected outright.** Cross-file discovery is the weakest category for all tools tested (max Recall@10 = 0.44 across competitors). Graph-lite metadata (imports, call edges) may help but the implementation cost is significant and the benefit is unproven. The core retrieval pipeline provides better ROI. See [ADR-000 Decision Summary](adr/000-decision-summary.md).

### ANN (Approximate Nearest Neighbor) Indexing
**Not needed at current scale.** sqlite-vec brute-force KNN at 10ms p50 on 5K+ chunks is fast enough. ANN (e.g., HNSW) adds complexity, memory overhead, and approximate (lossy) results without meaningful benefit under 100K chunks.

## 3. Open Risks

### API Dependency for Full Pipeline
Vera's full hybrid pipeline requires two external API round trips per query (embedding + reranker), adding 3–7 seconds of latency. **Mitigation:** BM25-only fallback is always available at sub-5ms latency; query embedding cache eliminates repeated embedding costs; Qwen3-0.6B is a viable local embedding model via Ollama. **Residual risk:** Air-gapped or latency-critical environments cannot use the full pipeline without deploying local models.

### sqlite-vec Maturity
sqlite-vec is v0.1.x with limited community adoption. **Mitigation:** The vector search surface area is small (insert, KNN query, delete); the module is easily replaceable if needed. SQLite itself is the most battle-tested database in existence. **Residual risk:** Potential bugs or performance regressions in sqlite-vec under edge cases.

### Cross-File Discovery Quality
The weakest retrieval category across all tools and all configurations (max Recall@10 = 0.44 for any competitor, 0.17 for Vera hybrid on the 17-task subset). **Mitigation:** Graph-lite metadata signals (imports, type references) are a known avenue for improvement. **Residual risk:** May require significant new infrastructure (import parsing, call graph) to meaningfully improve.

### Embedding Model API Cost and Availability
Indexing large repositories requires thousands of embedding API calls. Rate limits and per-token costs scale with repository size. **Mitigation:** Matryoshka truncation (4096→1024-dim) reduces storage by 4× without significant quality loss; batch processing reduces API calls; Qwen3-0.6B is a free alternative via local deployment. **Residual risk:** Cost unpredictability for very large monorepos.

### Turborepo Benchmark Gap
The turborepo polyglot repository (4 tasks) was excluded from final benchmarks due to persistent embedding API rate limits. **Mitigation:** All 5 workload categories are still represented in the 17-task subset. **Residual risk:** Vera's performance on very large polyglot monorepos is less thoroughly validated.

## 4. Next Steps

### Short-Term (v0.2)

1. **Local model support** — Integrate Ollama or similar for local embedding and reranking. This eliminates API dependency and reduces hybrid query latency from ~4s to ~50-100ms.

2. **Cross-file discovery improvement** — Add lightweight graph metadata: file-to-file import edges, symbol reference tracking. Target: cross-file Recall@10 > 0.50.

3. **Turborepo validation** — Complete benchmarks on the turborepo corpus with rate-limit mitigation (sequential indexing, extended cooldowns).

4. **Platform support** — Test and validate macOS builds via cross-compilation. Add CI for Linux + macOS release binaries.

### Medium-Term (v0.3)

5. **Weighted RRF tuning** — Experiment with learned or category-adaptive fusion weights (currently using standard RRF with k=60). May improve intent search where hybrid currently trails cocoindex-code.

6. **Intent search improvement** — Vera hybrid (MRR 0.46) trails cocoindex-code (MRR 0.63) on intent queries. Investigate: query expansion, multi-vector query encoding, or intent-specific retrieval paths.

7. **Incremental embedding updates** — Currently re-embeds all chunks in changed files. Implement chunk-level diffing to skip unchanged symbols within modified files.

8. **Configuration profiles** — Predefined configs for common use cases: `--profile fast` (BM25-only), `--profile balanced` (hybrid, no rerank), `--profile quality` (full pipeline).

### Long-Term

9. **ANN indexing** — If repository sizes exceed 100K chunks, add HNSW or IVF-PQ indexing to sqlite-vec or a dedicated vector store.

10. **Multi-repo indexing** — Support searching across multiple related repositories (monorepo sub-projects, microservice collections).

11. **Streaming results** — Return results as they become available (BM25 first, then vector, then reranked) for better perceived latency.

12. **Plugin system** — Allow custom chunking strategies, embedding providers, and output formatters without forking.
