# Architecture

Architectural decisions, patterns, and conventions for Vera.

**What belongs here:** Decided architecture patterns, module ownership, key design decisions.
**What does NOT belong here:** Speculative ideas (those go in ADRs until decided).

---

## Decided (updated as ADRs are finalized)

### ADR-001: Implementation Language → Rust
- Rust chosen over TypeScript/Bun based on spike benchmarks
- 1.6–1.8× faster tree-sitter parsing, 10× faster cold start vs Bun
- Single binary distribution (~10-15MB estimated) vs 32MB+ node_modules
- Superior ecosystem: `ignore` crate, Tantivy, Lance, Clap
- See `docs/adr/001-implementation-language.md` for full evidence

### ADR-002: Storage Backend → SQLite + sqlite-vec + Tantivy
- SQLite + sqlite-vec for metadata and vector search, Tantivy for BM25
- Chosen over LanceDB despite LanceDB's 5× faster vector queries and 32× faster writes
- SQLite performance is sufficient: 10ms vector query p50, 7.6K chunks/sec writes
- Key advantage: ~60 crates vs 537 (LanceDB), 40s vs 150s build time, sync API
- Tantivy BM25 is sub-millisecond (0.067ms p50) — uncontested for full-text search
- See `docs/adr/002-storage-backend.md` for full evidence

### ADR-003: Embedding Model → Qwen3-Embedding-8B
- Qwen3-Embedding-8B chosen over bge-en-icl and Qwen3-Embedding-0.6B
- Best Recall@10 (0.66), nDCG (0.71), and Recall@5 (0.49) on Vera's 21-task suite
- Outperforms all M1 competitor baselines on recall and nDCG
- 4096-dim vectors; OpenAI-compatible API via Nebius (EMBEDDING_MODEL_BASE_URL)
- Qwen3-Embedding-0.6B (1024-dim) designated as lightweight fallback for local use
- MRR (0.28) lags cocoindex-code (0.35), confirming reranking is essential
- See `docs/adr/003-embedding-model.md` for full evidence

### ADR-004: Chunking Strategy → Symbol-Aware (tree-sitter AST)
- Symbol-aware chunking chosen over sliding-window and file-level
- 2.3× higher MRR on symbol lookup (0.55 vs 0.24) — correct definitions rank ~2nd vs ~4th
- Best overall MRR (0.38 vs 0.28) and Recall@5 (0.59 vs 0.49)
- 14% more token-efficient than sliding-window (0.86 ratio)
- Intent search R@5=0.90 vs 0.50 for sliding-window
- Sliding-window as Tier 0 fallback for unsupported languages
- See `docs/adr/004-chunking-strategy.md` for full evidence

### ADR-000: Decision Summary
- All 5 major architecture questions decided: language, storage, embedding, chunking, retrieval pipeline shape
- Retrieval pipeline: BM25 (Tantivy) + Vector (sqlite-vec) → RRF fusion → Reranking (Qwen3-Reranker)
- 2 open questions remain: pipeline parameters (tune in M2), graph-lite scope (defer to M2/M3)
- 13 prior assumptions validated, 2 invalidated (LanceDB, sliding-window role), 7 hypothetical
- See `docs/adr/000-decision-summary.md` for full decision matrix and assumption categorization

## Baseline Findings from M1 Competitor Benchmarks

Key insights from competitor baseline benchmarking (21 tasks, 4 repos):

- **Lexical (ripgrep):** Recall@10=0.37, MRR=0.26, p50=18ms. Fast but poor semantic coverage.
- **Semantic (cocoindex-code):** Recall@10=0.50, MRR=0.35, p50=446ms. Balanced quality/speed.
- **Vector-only (Qwen3):** Recall@10=0.66, MRR=0.28, p50=1186ms. Best recall but slow and poor MRR.

**Design implications:**
1. Config lookup tasks are challenging for all tools. Consider filename/filetype matching stage.
2. Cross-file discovery is weak across all baselines (max Recall@10=0.44). Graph-lite metadata may help.
3. Hybrid BM25+vector is clearly justified: lexical is fast for identifiers, vector catches semantics.
4. Reranking is likely the key to MRR improvement over vector-only (high recall, poor MRR suggests ranking issue).

## Key Constraints

- Files under 300 lines (soft), 500 lines (hard - must explain)
- Functions under 40 lines (soft), 80 lines (hard - must explain)
- Explicit module ownership boundaries
- Side effects at boundaries only
- Composition over inheritance
- No magic-heavy patterns without justification
