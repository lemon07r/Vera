# Accuracy Improvements From `v0.4.0` To `v0.6.0`

This note explains why the current local benchmark numbers are much stronger than the older public comparisons that still appear in the README and benchmark docs.

The public comparison tables were recorded around the `v0.4.0` generation of Vera. Since then, the retrieval pipeline has been tightened across indexing, candidate generation, ranking, and evaluation. `v0.6.0` is the result of that work.

## Benchmark Summary

These three versions were measured with the same local-binary benchmark harness, the same pinned repositories, and the same local Jina CUDA ONNX stack.

- 21 tasks
- 4 repos: `ripgrep`, `flask`, `fastify`, `turborepo`
- hybrid retrieval with BM25, dense embeddings, and reranking

| Version | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|--------|----------|----------|-----------|--------|---------|
| `v0.4.0` | 0.2421 | 0.5040 | 0.5159 | 0.5016 | 0.4570 |
| `v0.5.0` | 0.3135 | 0.5635 | 0.6349 | 0.5452 | 0.5293 |
| `v0.6.0` | **0.8135** | **1.0000** | **1.0000** | **1.0000** | **0.9832** |

`Recall@1 = 0.8135` is the maximum possible score on this suite because several tasks have more than one ground-truth target.

Benchmark artifacts:

- [v0.4.0](/home/lamim/Development/Tools/Vera/benchmarks/results/local-binaries/v0.4.0-jina-cuda-onnx.json)
- [v0.5.0](/home/lamim/Development/Tools/Vera/benchmarks/results/local-binaries/v0.5.0-jina-cuda-onnx.json)
- [v0.6.0](/home/lamim/Development/Tools/Vera/benchmarks/results/local-binaries/v0.6.0-jina-cuda-onnx.json)

## What Changed

### Richer indexed text

Earlier versions relied too much on raw chunk content. The current pipeline feeds both BM25 and embeddings with more retrieval context:

- stronger filename and path signal
- better symbol naming
- file-level context for config and document-like files
- clearer structural chunks such as Rust `impl` blocks and Python class containers

This improved config lookup, symbol disambiguation, and cross-file search.

### Better exact-match behavior

Exact symbol lookup is much stricter than it was in `v0.4.0`.

- case-sensitive symbol matches are preserved
- exact type definitions are preferred over lowercase method names when the query asks for a type
- exported exact definitions are ranked ahead of shallow private duplicates
- crate-private items such as `pub(crate)` are no longer treated like public API definitions

This change is why ambiguous names such as `Config` now rank much more sensibly.

### Better structural and cross-file candidate selection

The search pipeline now does more work before final reranking.

- same-file structural context is pulled in when the top hit is too narrow
- related implementation blocks are added for queries such as `Sink trait and its implementations`
- cross-language concept matches are expanded when the same idea exists in more than one representation
- helper functions that complete an answer can be pulled in for intent-heavy queries
- candidate lists are diversified by file so one file does not crowd out the rest

This is what moved many tasks from "the right file appears somewhere" to "the full answer is already near the top".

### Stronger path and config ranking

Configuration queries benefit from more direct lexical handling than before.

- root and shallow config files get stronger path-based ranking
- filename matches matter more
- docs, tests, examples, and generated files are demoted unless the query clearly asks for them

That reduced the common failure mode where nested config files or incidental mentions outranked the real target.

### More stable local reranking

Local reranking is now batched more carefully. This avoids ONNX CUDA out-of-memory fallbacks that could silently reduce quality during local search and benchmarking.

### Cleaner evaluation

The benchmark harness also became more reliable.

- duplicate hits no longer inflate nDCG
- repeated overlaps against the same target are de-duplicated during scoring
- raw ranked results are saved for inspection

That makes the current benchmark numbers easier to trust than earlier internal runs.

## What This Means In Practice

The older public comparison tables in the README are still useful because they compare Vera against other tools on the same workload. They are not the best picture of Vera's current quality.

If you want the current state of the retrieval pipeline, use the `v0.6.0` local benchmark numbers in [docs/benchmarks.md](/home/lamim/Development/Tools/Vera/docs/benchmarks.md). If you want the historical comparison, keep the older public snapshot in view as a baseline.
