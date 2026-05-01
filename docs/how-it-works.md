# How Vera Works

Vera's search pipeline has three stages: retrieve candidates, fuse results, then rerank. Every stage was chosen based on benchmarks against real codebases, not assumptions.

## Parsing: Tree-Sitter Chunks

Vera parses source files into ASTs using tree-sitter grammars compiled into the binary. Instead of splitting code into arbitrary line ranges, it extracts discrete structural units such as functions, classes, structs, traits, interfaces, methods, and `impl` blocks.

For config and document-like files, Vera uses whole-file chunks instead. Module-level gaps between symbols are also kept as chunks when they carry useful retrieval context.

Each chunk carries metadata: file path, line range, language, symbol name, and symbol type. This means search results map to actual code boundaries, not random slices.

Large symbols (>150 lines) are split at logical boundaries. Languages without a tree-sitter grammar fall back to sliding-window chunking. See [features.md](features.md#tree-sitter-structural-parsing) for chunking benchmarks.

During parsing, Vera also records file-level diagnostics such as tree-sitter error nodes, Tier 0 fallback, and outright parse failures. `vera stats` surfaces these later as index-health signals instead of silently dropping them on the floor.

## Retrieval: BM25 + Vector Search

Two retrieval paths run in parallel for every query:

**BM25 (keyword matching)** uses a Tantivy index over structured chunk text, including content, symbol names, file paths, and filename/path tokens. It handles exact identifier and config-style lookups. searching for `parse_config` finds that exact function. BM25 alone scores sub-millisecond latency (0.067ms p50).

**Vector search (semantic matching)** embeds the query and compares it against pre-computed chunk embeddings stored in sqlite-vec. This catches conceptual matches. searching "authentication middleware" finds relevant auth code even if those exact words don't appear. Vector search alone achieves 0.66 Recall@10 but only 0.28 MRR@10 (high recall, poor ranking).

Neither path alone is sufficient. BM25 misses semantic matches. Vector search misses exact identifiers and ranks poorly. Combining them covers both.

## Fusion: Reciprocal Rank Fusion

Results from both retrieval paths are merged using Reciprocal Rank Fusion (RRF). RRF scores each result based on its rank in each list:

```
score(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_vector(d))
```

A result that ranks high in both lists gets a high fused score. A result that ranks high in only one list still appears, but lower. The constant `k` (default: 60) controls how much weight goes to top-ranked vs. lower-ranked results.

RRF is simple, parameter-light, and doesn't need training data. It consistently outperforms either retrieval path alone.

## Query-Aware Ranking

After fusion, Vera applies lightweight deterministic ranking logic before final reranking.

This stage handles cases that dense retrieval alone is bad at:

- exact filename and exact identifier queries
- path-heavy config lookups
- noisy test and docs matches
- broad natural-language queries that need structural results instead of tiny helpers
- same-file and cross-file answer completion

For broad intent queries, Vera also keeps a deeper fused candidate pool before final truncation. This matters when raw RRF pushes the right `struct` or `impl` block just outside the requested top N and a tiny helper would otherwise win by default.

This is also where Vera adds a small amount of query-aware candidate expansion, such as pulling in related implementation blocks or same-file structural context when the initial hit is too narrow.

## Reranking: Cross-Encoder

The top fused candidates are sent to a cross-encoder reranker. Unlike embeddings (which encode query and document separately), the cross-encoder reads the query and each candidate together as a single pair, scoring relevance jointly.

This is the most expensive stage but also the most impactful. Reranking lifts MRR@10 from 0.39 to 0.60, a 54% improvement in how often the best result appears at the top.

With Jina ONNX local models, the reranker runs on-device via ONNX Runtime. Potion Code uses deterministic ranking heuristics instead of the ONNX reranker. With API mode, reranking calls your configured endpoint. Obvious filename and path-dominant queries can skip reranking when lexical evidence is already decisive.

Large candidate sets are batched automatically to stay within the reranker's request limits. Oversized documents are truncated at newline boundaries before scoring. See [features.md](features.md#cross-encoder-reranking) for configuration details.

## Storage

Everything lives in two places:

- **`.vera/`** in the project root. SQLite metadata (chunks, file hashes, file-level index state), Tantivy BM25 index, and sqlite-vec vector store. One directory per project.
- **`$XDG_DATA_HOME/vera/models/`** (or `~/.vera/models/` on existing installs): cached local model assets. Downloaded once by `vera setup`.

The index is a single SQLite database file plus a Tantivy directory. No external services, no daemons, no background processes.

## Incremental Updates

`vera update .` compares content hashes against stored metadata, re-processes only changed files, and refreshes the persisted file-level index state. That keeps parse failures and fallback counts visible across incremental runs. See [features.md](features.md#incremental-updates) for details.

## Pipeline Summary

```
Query
  ├─→ BM25 search (Tantivy)        ──→ ranked candidates
  └─→ Vector search (sqlite-vec)    ──→ ranked candidates
                                          │
                                    RRF fusion
                                          │
                             query-aware ranking and expansion
                                          │
                                    top candidates
                                          │
                                    cross-encoder rerank
                                          │
                                    final ranked results
```

| Stage | What it does | Why it matters |
|-------|-------------|----------------|
| Tree-sitter parsing | Extracts symbols as chunks | Results map to real code boundaries |
| BM25 | Exact keyword matching | Catches identifiers, fast |
| Vector search | Semantic similarity | Catches conceptual matches |
| RRF fusion | Merges both result lists | Covers both exact and semantic |
| Query-aware ranking | Applies deterministic priors and candidate shaping | Fixes exact-match, config, and cross-file failure modes |
| Cross-encoder rerank | Joint query-document scoring | Best result lands at the top |
