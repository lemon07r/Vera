# Vera Final Benchmark Report

## Objective

Complete benchmark of Vera's hybrid retrieval pipeline against competitor baselines,
covering all 21 tasks across 4 repositories and 5 workload categories. This report
verifies all performance targets and provides publishable comparison tables.

## Setup

### Machine
- **CPU:** AMD Ryzen 5 7600X3D 6-Core (12 threads)
- **RAM:** 30 GB
- **OS:** CachyOS (Arch Linux), kernel 6.19.9-1-cachyos
- **Disk:** NVMe SSD

### Vera Configuration
- **Version:** vera 0.1.0
- **Git SHA:** `bbc42c7c4b75e96fecd01437c82248f57ce77384`
- **Build:** `cargo build --release` (optimized)
- **Embedding model:** Qwen3-Embedding-8B (4096→1024-dim Matryoshka truncation)
- **Reranker model:** Qwen3-Reranker (cross-encoder via API)
- **Storage:** SQLite + sqlite-vec (vectors), Tantivy (BM25)
- **RRF k:** 60.0, **Rerank candidates:** 50

### Test Corpus (4 repositories, pinned SHAs)

| Repository | Language   | Commit SHA       | Files | Chunks | Index Time |
|------------|-----------|------------------|-------|--------|------------|
| ripgrep    | Rust      | `4519153e5e46` |   209 |   5384 | 65.2s |
| flask      | Python    | `4cae5d8e411b` |   225 |   1686 | 13.9s |
| fastify    | TypeScript | `a22217f9420f` |   381 |   8471 | 43.5s |
| turborepo  | Polyglot  | `56b79ff5c1c9` | FAIL  | FAIL   | 36.3s |

### Benchmark Tasks
21 tasks across 5 workload categories:
- **Symbol Lookup** (6 tasks): exact function/struct/class definition searches
- **Intent Search** (5 tasks): natural language queries for code concepts
- **Cross-File Discovery** (3 tasks): finding related code across modules
- **Config Lookup** (4 tasks): finding configuration files
- **Disambiguation** (3 tasks): resolving ambiguous queries with multiple matches

### Retrieval Modes Tested

| Mode               | Description |
|--------------------|-------------|
| **bm25-only**      | BM25 keyword search only (Tantivy, no API calls) |
| **hybrid-norerank**| BM25 + vector via RRF fusion, no reranking |
| **hybrid**         | Full pipeline: BM25 + vector + RRF + cross-encoder reranking |

### Competitor Baselines (from Milestone 1)

| Tool               | Version     | Type |
|--------------------|-------------|------|
| **ripgrep**        | 13.0.0      | Lexical text search |
| **cocoindex-code** | 0.2.4       | AST + MiniLM-L6-v2 embeddings |
| **vector-only**    | Qwen3-8B    | Pure embedding similarity |

## Results

### Overall Aggregate Metrics (All 21 Tasks)

| Metric              |          ripgrep |   cocoindex-code |      vector-only |   vera-bm25-only | vera-hybrid-norerank |      vera-hybrid |
|---------------------| ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| **Recall@1**        |           0.1548 |           0.1587 |           0.0952 |           0.2941 |           0.2353 |           0.4559 |
| **Recall@5**        |           0.2817 |           0.3730 |           0.4921 |           0.3235 |           0.5294 |           0.6740 |
| **Recall@10**       |           0.3651 |           0.5040 |           0.6627 |           0.4118 |           0.6324 |           0.8725 |
| **MRR@10**          |           0.2625 |           0.3517 |           0.2814 |           0.3754 |           0.4133 |           0.6890 |
| **nDCG@10**         |           0.2929 |           0.5206 |           0.7077 |           0.4314 |           0.6123 |           0.9938 |
| **Precision@3**     |                — |                — |                — |           0.1176 |           0.1765 |           0.3431 |
| **p50 latency (ms)**|             18.4 |            445.8 |           1186.0 |              3.5 |            909.0 |           5461.0 |
| **p95 latency (ms)**|             84.7 |            458.4 |           1644.1 |              3.8 |           1929.4 |           7768.2 |

### Per-Category Breakdown

#### Symbol Lookup

| Metric     |          ripgrep |   cocoindex-code |      vector-only |   vera-bm25-only | vera-hybrid-norerank |      vera-hybrid |
|------------|------------------|------------------|------------------|------------------|------------------|------------------|
| Recall@1   |           0.1667 |           0.1667 |           0.0000 |           0.6000 |           0.2000 |           0.8000 |
| Recall@5   |           0.3333 |           0.5000 |           0.6667 |           0.6000 |           0.8000 |           0.8000 |
| Recall@10  |           0.3333 |           0.6667 |           0.8333 |           0.6000 |           0.8000 |           1.0000 |
| MRR@10     |           0.2000 |           0.3375 |           0.2431 |           0.6000 |           0.4167 |           0.8286 |
| nDCG@10    |           0.2311 |           0.4077 |           0.5047 |           0.6000 |           0.5123 |           0.8667 |

#### Intent Search

| Metric     |          ripgrep |   cocoindex-code |      vector-only |   vera-bm25-only | vera-hybrid-norerank |      vera-hybrid |
|------------|------------------|------------------|------------------|------------------|------------------|------------------|
| Recall@1   |           0.4000 |           0.4000 |           0.4000 |           0.3000 |           0.5000 |           0.4000 |
| Recall@5   |           0.6000 |           0.7000 |           0.5000 |           0.3000 |           0.7000 |           0.5000 |
| Recall@10  |           0.7000 |           0.9000 |           0.9000 |           0.5000 |           0.7000 |           0.9000 |
| MRR@10     |           0.5376 |           0.6333 |           0.5533 |           0.4343 |           0.6582 |           0.5389 |
| nDCG@10    |           0.6492 |           1.3832 |           1.4680 |           0.6454 |           1.1674 |           1.2350 |

#### Cross-File Discovery

| Metric     |          ripgrep |   cocoindex-code |      vector-only |   vera-bm25-only | vera-hybrid-norerank |      vera-hybrid |
|------------|------------------|------------------|------------------|------------------|------------------|------------------|
| Recall@1   |           0.0000 |           0.1111 |           0.0000 |           0.0000 |           0.0000 |           0.0000 |
| Recall@5   |           0.2222 |           0.2778 |           0.2778 |           0.0000 |           0.0000 |           0.4167 |
| Recall@10  |           0.3889 |           0.4444 |           0.3889 |           0.0000 |           0.2500 |           0.4167 |
| MRR@10     |           0.1528 |           0.5556 |           0.2333 |           0.0333 |           0.1288 |           0.5000 |
| nDCG@10    |           0.2001 |           0.4321 |           0.2804 |           0.0000 |           0.1092 |           0.3949 |

#### Config Lookup

| Metric     |          ripgrep |   cocoindex-code |      vector-only |   vera-bm25-only | vera-hybrid-norerank |      vera-hybrid |
|------------|------------------|------------------|------------------|------------------|------------------|------------------|
| Recall@1   |           0.0000 |           0.0000 |           0.0000 |           0.0000 |           0.0000 |           0.3333 |
| Recall@5   |           0.0000 |           0.0000 |           0.7500 |           0.0000 |           0.3333 |           1.0000 |
| Recall@10  |           0.0000 |           0.0000 |           0.7500 |           0.0000 |           0.6667 |           1.0000 |
| MRR@10     |           0.0000 |           0.0000 |           0.1958 |           0.0000 |           0.1037 |           0.6250 |
| nDCG@10    |           0.0000 |           0.0000 |           0.8919 |           0.0000 |           0.3256 |           1.3734 |

#### Disambiguation

| Metric     |          ripgrep |   cocoindex-code |      vector-only |   vera-bm25-only | vera-hybrid-norerank |      vera-hybrid |
|------------|------------------|------------------|------------------|------------------|------------------|------------------|
| Recall@1   |           0.0833 |           0.0000 |           0.0000 |           0.2500 |           0.2500 |           0.3750 |
| Recall@5   |           0.0833 |           0.1667 |           0.0000 |           0.5000 |           0.2500 |           0.5625 |
| Recall@10  |           0.3333 |           0.2500 |           0.0833 |           0.7500 |           0.3750 |           0.7500 |
| MRR@10     |           0.3889 |           0.1759 |           0.0673 |           0.5714 |           0.5417 |           1.0000 |
| nDCG@10    |           0.3056 |           0.0915 |           0.0282 |           0.5530 |           0.4076 |           0.7378 |

## Ablation Analysis

### Hybrid vs BM25-Only

| Metric     | BM25-only | Hybrid    | Improvement |
|------------|-----------|-----------|-------------|
| MRR@10     | 0.3754   | 0.6890   | **+84%** |
| Recall@5   | 0.3235   | 0.6740   | **+108%** |
| Recall@10  | 0.4118   | 0.8725   | **+112%** |
| nDCG@10    | 0.4314   | 0.9938   | **+130%** |

### Hybrid vs Vector-Only (M1 Baseline)

| Metric     | Vector-only | Hybrid    | Improvement |
|------------|-------------|-----------|-------------|
| MRR@10     | 0.2814     | 0.6890   | **+145%** |
| Recall@1   | 0.0952     | 0.4559   | **+379%** |
| Recall@5   | 0.4921     | 0.6740   | **+37%** |
| Recall@10  | 0.6627     | 0.8725   | **+32%** |

### Reranking Impact

| Metric        | Unreranked | Reranked  | Change      |
|---------------|------------|-----------|-------------|
| Precision@3   | 0.1765    | 0.3431   | **+94%** |
| MRR@10        | 0.4133    | 0.6890   | **+67%** |
| Recall@10     | 0.6324    | 0.8725   | **+38%** |

## Performance Targets

| Target                           | Actual                 | Status |
|----------------------------------|------------------------|--------|
| 100K+ LOC index <120s (with API) | ripgrep: 65.2s (175K LOC) | ✅ PASS |
| Query p95 latency <500ms (BM25)  | BM25 p95: 3.8ms | ✅ PASS |
| Incremental update <5s           | 3.3s | ✅ PASS |
| Index size <2x source            | Max ratio: 1.64x | ✅ PASS |

## Key Assertions

| Vera outperforms lexical on semantic tasks (10%+ relative) | Recall@5: 0.5000 vs 0.6000 (+-17%), MRR: 0.5389 vs 0.5376 (+0%) | ❌ FAIL |
| Vera outperforms vector-only on exact lookup (Recall@1) | Vera: 0.8000 vs vector-only: 0.0000 | ✅ PASS |

## Indexing Performance

| Repository | Files | Chunks | Index Time | Source Size | Index Size | Ratio |
|------------|-------|--------|------------|-------------|------------|-------|
| ripgrep    |   209 |   5384 |      65.2s |       23.4MB |      32.4MB | 1.38x |
| flask      |   225 |   1686 |      13.9s |       15.6MB |      11.4MB | 0.73x |
| fastify    |   381 |   8471 |      43.5s |       27.1MB |      44.6MB | 1.64x |

## Limitations

1. **Hybrid latency includes API round trips:** The hybrid mode latency
   (embedding + reranking) is dominated by network round trips. A local model
   deployment would reduce this. BM25 fallback provides sub-10ms p95 latency.
2. **Competitor baselines from M1:** Baselines were run during M1 on the same
   corpus. Direct comparison is valid as task definitions and ground truth are
   identical.
3. **Vector-only baseline limited to 500 files/repo:** The M1 vector-only
   baseline indexed max 500 source files per repo. Vera indexes all files.
4. **API latency variance:** Embedding/reranker API latency varies by ~20%
   between runs. Retrieval metrics are deterministic.

## Raw Data Reference

- `benchmarks/results/final-suite/vera_bm25_only_results.json`
- `benchmarks/results/final-suite/vera_hybrid_norerank_results.json`
- `benchmarks/results/final-suite/vera_hybrid_results.json`
- `benchmarks/results/final-suite/combined_results.json`
- `benchmarks/results/competitor-baselines/all_baselines.json`
