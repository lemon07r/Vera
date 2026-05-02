# Vera Benchmarks

This page tracks two benchmark snapshots:

- the current local release benchmark used to tune retrieval quality
- the older public API benchmark kept for historical comparison

## Current Local Release Benchmark

This is the benchmark used to measure the `v0.7.0` retrieval pipeline.

- 21 tasks
- 4 repos: `ripgrep`, `flask`, `fastify`, `turborepo`
- local Jina embedding + reranker stack
- CUDA ONNX backend
- same pinned corpora and the same local-binary harness for every version below

### Accuracy Improvements From `v0.4.0` To `v0.7.0`

| Version | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|--------|----------|----------|-----------|--------|---------|
| `v0.4.0` | 0.2421 | 0.5040 | 0.5159 | 0.5016 | 0.4570 |
| `v0.5.0` | 0.3135 | 0.5635 | 0.6349 | 0.5452 | 0.5293 |
| `v0.7.0` | **0.7183** | **0.7778** | **0.8254** | **0.9095** | **0.8361** |

From `v0.4.0` to `v0.7.0`, Vera improved by:

- `+0.4762` Recall@1
- `+0.2738` Recall@5
- `+0.3095` Recall@10
- `+0.4079` MRR@10
- `+0.3791` nDCG@10

Committed artifacts:

- [v0.4.0 benchmark](../benchmarks/results/local-binaries/v0.4.0-jina-cuda-onnx.json)
- [v0.5.0 benchmark](../benchmarks/results/local-binaries/v0.5.0-jina-cuda-onnx.json)
- [v0.7.0 benchmark](../benchmarks/results/local-binaries/v0.7.0-jina-cuda-onnx.json)
- [`v0.7.0` accuracy improvements](./releases/v0.7.0-accuracy-improvements.md)

### Current Performance Snapshot

`v0.7.0` local Jina CUDA ONNX results:

| Measure | Result |
|---------|--------|
| Search latency p50 | `3716 ms` |
| Search latency p95 | `4185 ms` |

### Recent Local Tuning Loop

These runs were used for retrieval tuning after `v0.7.0`. They are useful for regression tracking, but they are not the public release snapshot above.

Method:

- `benchmarks/scripts/run_local_binary_benchmarks.py` against the same 21-task, 4-repo corpus
- forced model paths on CUDA so quantized and fp16 runs did not silently switch models
- judged by the full suite, not Vera usage rate or one benchmark hole

Artifacts:

- pre-fix quantized: [c7bdc09-jina-cuda-onnx-quantized-embed](../benchmarks/results/local-binaries/c7bdc09-jina-cuda-onnx-quantized-embed.json)
- pre-fix fp16: [c7bdc09-jina-cuda-onnx-fp16-embed](../benchmarks/results/local-binaries/c7bdc09-jina-cuda-onnx-fp16-embed.json)
- current fp16 candidate-pool fix: [candidate-pool-fix-rerank50-jina-cuda-onnx-fp16-embed](../benchmarks/results/local-binaries/candidate-pool-fix-rerank50-jina-cuda-onnx-fp16-embed.json)
- current quantized candidate-pool fix: [oom-fix-jina-cuda-onnx-quantized-embed](../benchmarks/results/local-binaries/oom-fix-jina-cuda-onnx-quantized-embed.json)
- current quantized dynamic scaler: [dynamic-scaler-jina-cuda-onnx-quantized-embed](../benchmarks/results/local-binaries/dynamic-scaler-jina-cuda-onnx-quantized-embed.json)

Current fp16 candidate-pool fix vs pre-fix fp16:

| Metric | Pre-fix fp16 | Current fp16 |
|--------|--------------|--------------|
| Recall@1 | 0.7183 | **0.7659** |
| Recall@5 | 0.8254 | **0.8968** |
| Recall@10 | 0.8254 | **0.8968** |
| MRR@10 | 0.9206 | **0.9683** |
| nDCG@10 | 0.8425 | **0.9027** |

What changed:

- `intent-004` (`file type detection and filtering`) moved from a miss to a perfect hit by returning `crates/ignore/src/types.rs:224-301` instead of a tiny helper method
- `cross-file-002` improved because the deeper pool also kept the second relevant blueprint registration chunk alive long enough to rank
- no task regressed in the fp16 full-suite rerun

Tradeoff:

- search latency went up on the fp16 tuning run (`p50 4103 ms`, `p95 10772 ms`)
- most of the extra cost came from broad intent queries that now search a deeper candidate pool before truncation

Quantized note:

- the full forced-quantized 21-task rerun now completes and matches the fp16 aggregate metrics on this suite
- on this machine, quantized ended up slightly faster on search (`p50 3617 ms` vs `4103 ms`) but slower on indexing
- the original blocker was a large `turborepo` embedding batch that hit a CUDA ONNX allocation spike inside `MultiHeadAttention`; Vera now retries those local batches at smaller sizes instead of aborting the index
- the dynamic sequence-aware scaler keeps the same aggregate metrics as `oom-fix-jina-cuda-onnx-quantized-embed`, then trims quantized indexing time on every benchmark repo in the same 21-task run (`ripgrep 13.08s -> 12.96s`, `fastify 15.24s -> 14.57s`, `turborepo 55.28s -> 54.71s`, `flask 6.37s -> 5.82s`)
- the scaler now also persists learned GPU windows across runs in `~/.vera/adaptive-batch-scaler.json`; when you compare cold indexing throughput, clear that file first or run all candidates against the same warmed state

### Semble Comparison

Vera is benchmarked against [Semble](https://github.com/MinishLab/semble), a Python code search tool using `potion-code-16M` static embeddings.

**320-task subset** (16 repos, used for tuning iteration):

| Tool | Backend | Recall@1 | Recall@10 | MRR | nDCG@10 | Search p50 | Search p95 | Index time |
|------|---------|----------|-----------|-----|---------|------------|------------|------------|
| Semble | Potion Code CPU | 0.6630 | 0.9479 | 0.8223 | **0.8311** | **1.43 ms** | **15.41 ms** | 26.06 s |
| Vera | BM25 ranked (v2) | 0.5542 | 0.8750 | 0.7349 | 0.7456 | 3.24 ms | 11.62 ms | **10.28 s** |
| Vera | BM25 ranked (v1) | 0.5214 | 0.8438 | 0.6949 | 0.7108 | 2.91 ms | 9.28 ms | 12.51 s |
| Vera | Potion Code CPU | 0.5010 | 0.8500 | 0.6700 | 0.6944 | 14.30 ms | 53.27 ms | 17.27 s |
| Vera | Jina CUDA ONNX | 0.5276 | 0.8578 | 0.7058 | 0.7233 | 23.50 ms | 6236.60 ms | 151.20 s |

v1 = pre-improvement baseline. v2 = English stemming, stronger definition boost, concept-to-filename augmentation.

**Full 1,251-task Semble suite** (63 repos, gate for parity claims):

| Metric | Vera BM25 (v2) |
|--------|---------------:|
| nDCG@10 | 0.6995 |
| Recall@1 | 0.5345 |
| Recall@10 | 0.8116 |
| MRR | 0.6838 |
| Search p50 | 4.03 ms |
| Search p95 | 19.97 ms |

Per-category: symbol_lookup 0.8955, intent 0.6873, cross_file 0.6080. Cross-file is the weakest category since many multi-hop queries need semantic understanding beyond BM25.

The Jina CUDA run uses CUDA ONNX Runtime via `ORT_DYLIB_PATH`. Do not run this lane against the CPU ONNX Runtime when comparing latency.

Artifacts:

- [Semble subset baseline](../benchmarks/results/semble/2026-04-29-semble-subset.json)
- [Vera BM25 subset v1](../benchmarks/results/semble/2026-05-01-vera-bm25-subset.json)
- [Vera BM25 subset v2](../benchmarks/results/semble/2026-05-02-vera-bm25-subset.json)
- [Vera BM25 full suite](../benchmarks/results/semble/2026-05-02-vera-bm25-full.json)
- [Vera Potion subset](../benchmarks/results/semble/2026-05-01-vera-potion-subset.json)
- [Vera Jina CUDA subset](../benchmarks/results/semble/2026-05-01-vera-cuda-subset.json)

### Optional CodeRankEmbed Preset

Vera now ships CodeRankEmbed as an optional local embedding preset. This is the short no-rerank sanity check used to decide whether it was worth exposing as a first-class option:

- 6 tasks
- 2 repos: `flask`, `ripgrep`
- local CUDA ONNX backend
- reranking disabled on purpose to expose embedding differences directly

| Model | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG | Search p50 | Flask index | Ripgrep index |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jina preset | 0.5556 | 0.5556 | 0.5556 | 0.8462 | 0.6442 | 761.9 ms | 5.8 s | 11.9 s |
| CodeRankEmbed preset | 0.7222 | 0.7222 | 0.7222 | 1.0000 | 0.8108 | 611.4 ms | 14.7 s | 29.1 s |

Takeaway: CodeRankEmbed was clearly stronger on this small no-rerank slice, but it indexed much slower. The default local benchmark and docs still center Jina because Vera's full reranked pipeline is already very strong and the shorter indexing time matters in practice.

## Vera vs ColGREP

These ColGREP numbers are the earlier reference results recorded on the same 21-task, 4-repo suite. They remain useful as a retrieval quality reference because they show how the current Vera pipeline compares with a late-interaction code search system on the same workload.

| Metric | Vera `v0.7.0` | ColGREP (149M) | ColGREP Edge (17M) |
|--------|---------------|----------------|--------------------|
| Recall@1 | **0.7183** | 0.5710 | 0.5240 |
| Recall@5 | **0.7778** | 0.6670 | 0.5710 |
| Recall@10 | **0.8254** | 0.7140 | 0.7140 |
| MRR@10 | **0.9095** | 0.6170 | 0.5660 |
| nDCG@10 | **0.8361** | 0.5610 | 0.5240 |

Indexing time, 4 repos combined:

| Tool | Total time | Hardware |
|------|-----------|----------|
| Vera `v0.7.0` | `~70 s` | RTX 4080 |
| ColGREP (149M, CPU) | `~180 s` | Ryzen 5 7600X3D 6c/12t |
| ColGREP Edge (17M, CPU) | `~160 s` | Ryzen 5 7600X3D 6c/12t |

ColGREP's late-interaction design was a useful reference while improving Vera's own ranking and chunk selection.

## Legacy Public API Benchmark

This is the older public benchmark snapshot that still appears in older docs and release notes.

- 17 tasks
- 3 repos: `ripgrep`, `flask`, `fastify`
- mixed API and local runs

### Retrieval Quality

| Metric | ripgrep | cocoindex-code | vector-only | Vera hybrid |
|--------|---------|----------------|-------------|-------------|
| Recall@1 | 0.1548 | 0.1587 | 0.0952 | **0.4265** |
| Recall@5 | 0.2817 | 0.3730 | 0.4921 | **0.6961** |
| Recall@10 | 0.3651 | 0.5040 | 0.6627 | **0.7549** |
| MRR@10 | 0.2625 | 0.3517 | 0.2814 | **0.6009** |
| nDCG@10 | 0.2929 | 0.5206 | 0.7077 | **0.8008** |

### Local vs API Models

The local Jina models were competitive with the much larger Qwen3-Embedding-8B API model on that older 17-task benchmark:

| Metric | Jina local (ONNX) | Qwen3-8B (API) |
|--------|-------------------|----------------|
| MRR@10 | **0.68** | 0.60 |
| Recall@5 | 0.65 | **0.73** |
| Recall@10 | 0.73 | **0.75** |
| nDCG@10 | 0.72 | **0.81** |

### Performance Snapshot

From the same older benchmark set:

| Measure | Result |
|---------|--------|
| BM25-only search p95 | `3.5 ms` |
| Hybrid search p95 | `6749 ms` |
| `ripgrep` index time | `65.1 s` |
| `flask` index time | `20.2 s` |
| `fastify` index time | `41.8 s` |

## Limits And Caveats

- The current release benchmark is deterministic and fully local, which makes it better for regression gating.
- The legacy public snapshot is still useful for older comparisons, but it should not be treated as the current retrieval baseline.
- Benchmark numbers in this repository show comparative behavior, not a promise that another machine or codebase will land on the same values.

## Related Docs

- [`v0.7.0` accuracy improvements](./releases/v0.7.0-accuracy-improvements.md)
- [Indexing performance note](../benchmarks/indexing-performance.md)
- [Reproduction guide](../benchmarks/reports/reproduction-guide.md)
