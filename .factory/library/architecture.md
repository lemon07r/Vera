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
