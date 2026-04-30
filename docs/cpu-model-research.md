# CPU-Friendly Models for Code Search: Research Report

## Context

Vera currently ships with:
- **Embedding**: jinaai/jina-embeddings-v5-text-nano-retrieval (768 dim, quantized INT8 ONNX)
- **Reranker**: jinaai/jina-reranker-v2-base-multilingual (quantized ONNX)

Competitor "semble" achieves NDCG@10 of 0.854 using model2vec/potion-code-16M (16M params, static embeddings, no transformer) with ~250ms indexing, ~1.5ms queries, all on CPU.

The goal: find embedding + reranker combos that deliver sub-100ms query latency on CPU while maintaining good code search quality.

---

## Embedding Models Comparison

| Model | Params | Dim | ONNX? | CPU Latency (per query) | Code-Specific? | Architecture | License |
|---|---|---|---|---|---|---|---|
| **model2vec/potion-code-16M** | 16M | 256 | Not needed (safetensors matrix lookup) | **<0.1ms** | Yes (distilled from CodeRankEmbed) | Static embeddings: token lookup + mean pool. No transformer. | MIT |
| **Snowflake/arctic-embed-xs** | 22M | 384 | Yes (community export) | ~5-15ms | No (general text) | MiniLM-L6-v2 based, 6 layers | Apache 2.0 |
| **Snowflake/arctic-embed-s** | 33M | 384 | Yes | ~10-20ms | No (general text) | MiniLM-L12-v2 based | Apache 2.0 |
| **nomic-ai/nomic-embed-text-v1.5** | 137M | 768 (Matryoshka: 64-768) | Yes | ~30-60ms | No (general text) | BERT-based, 8192 ctx | Apache 2.0 |
| **nomic-ai/CodeRankEmbed** | 137M | 768 | Exportable | ~30-60ms | Yes (code-specific, high quality) | BERT-based | Apache 2.0 |
| **nomic-ai/nomic-embed-code** | 137M | 768 | Exportable | ~30-60ms | Yes (state-of-the-art code retrieval, Mar 2025) | nomic-embed fine-tuned for code | Apache 2.0 |
| **jina-embeddings-v5-text-nano** (current) | ~30M | 768 | Yes (INT8) | ~15-30ms | No (general) | Transformer | Apache 2.0 |

### Key Embedding Findings

**model2vec/potion-code-16M is the clear winner for CPU speed:**
- Architecture: No transformer forward pass at all. It distills a large embedding model (CodeRankEmbed, 137M) into a static word embedding matrix. At inference time, it tokenizes the input, looks up each token's embedding vector in a matrix, and averages them. This is a simple matrix index + mean operation.
- Speed: ~500x faster than transformer models. Sub-millisecond per embedding.
- Has an **official Rust crate** (`model2vec-rs` on crates.io) for native Rust inference. No ONNX runtime needed.
- Supports safetensors format with f32, f16, and i8 weight types.
- Specifically trained for code retrieval (distilled from CodeRankEmbed on cornstack code datasets covering Python, Java, PHP, Go, JavaScript, Ruby).
- Created April 2026, very recent.
- 256-dimensional embeddings (smaller vectors = faster similarity search).

**Trade-off**: Static embeddings sacrifice context-sensitivity. All occurrences of a token get the same vector regardless of surrounding context. This hurts on nuanced queries but works well enough for code search (where identifiers, keywords, and structural patterns dominate).

---

## Reranker Models Comparison

| Model | Params | ONNX? | CPU Latency (per pair) | Quality (MS MARCO MRR@10) | Layers / Hidden | Context | License |
|---|---|---|---|---|---|---|---|
| **cross-encoder/ms-marco-TinyBERT-L2-v2** | **4.4M** | **Yes** (official) | **~1-3ms** | ~27 | 2L / 128H | 512 | Apache 2.0 |
| **cross-encoder/ms-marco-MiniLM-L2-v2** | 15.6M | Yes (official) | ~3-8ms | ~32 | 2L / 384H | 512 | Apache 2.0 |
| **jinaai/jina-reranker-v1-tiny-en** | ~33M | Yes (llmware ONNX export) | ~5-15ms | Good (Jina benchmark) | 4L / jina-bert | **8192** | Apache 2.0 |
| **cross-encoder/ms-marco-MiniLM-L6-v2** | 22.7M | Yes | ~10-25ms | ~34 | 6L / 384H | 512 | Apache 2.0 |
| **BAAI/bge-reranker-base** | 278M | Exportable | ~50-100ms | High | 12L / XLM-R | 512 | MIT |
| **BAAI/bge-reranker-v2-m3** | 568M | Yes (community) | ~200-500ms | Very high | 24L / XLM-R | 8192 | MIT |
| **jina-reranker-v2-base-multilingual** (current) | ~278M | Yes (quantized) | ~50-100ms | Very high | 12L | 8192 | Apache 2.0 |

### Key Reranker Findings

**cross-encoder/ms-marco-TinyBERT-L2-v2 is the fastest viable reranker:**
- Only 4.4M parameters (2 layers, 128 hidden dim). Base model: nreimers/BERT-Tiny_L-2_H-128_A-2.
- Official ONNX export available on HuggingFace (tagged `onnx`).
- 23M+ all-time downloads. Extremely well-tested in production.
- At ~1-3ms per query-document pair on CPU, you can rerank 20 candidates in 20-60ms.
- Quality is lower than larger models but still provides meaningful reranking signal.

**cross-encoder/ms-marco-MiniLM-L2-v2 is the quality-step-up option:**
- 15.6M parameters (2 layers, 384 hidden dim). Same 2-layer architecture but wider.
- Still fast enough at ~3-8ms per pair (20 candidates in 60-160ms).
- Better ranking quality than TinyBERT.

**jina-reranker-v1-tiny-en is interesting for code (8K context):**
- 8192 token context window, which matters for code chunks that can be long.
- ~33M params, 4 layers. ONNX available via llmware export.
- Slightly slower but handles longer code snippets without truncation.

---

## Recommended Combinations for Vera CPU Mode

### Option A: Maximum Speed (semble-competitive)
- **Embedding**: model2vec/potion-code-16M (via model2vec-rs Rust crate, no ONNX needed)
- **Reranker**: cross-encoder/ms-marco-TinyBERT-L2-v2 (4.4M, ONNX)
- **Expected latency**: <1ms embed + ~40ms rerank (20 candidates) = **~40ms total**
- **Quality**: Good for code search. Static embeddings recall + transformer reranking quality.
- **Model size on disk**: ~64MB (16M embed weights) + ~17MB (4.4M reranker ONNX) = **~81MB**
- **Integration notes**: model2vec-rs is a native Rust crate on crates.io. No ONNX for embeddings. Only ONNX needed for the reranker.

### Option B: Better Quality, Still Fast
- **Embedding**: model2vec/potion-code-16M (via model2vec-rs)
- **Reranker**: cross-encoder/ms-marco-MiniLM-L2-v2 (15.6M, ONNX)
- **Expected latency**: <1ms embed + ~100ms rerank (20 candidates) = **~100ms total**
- **Quality**: Better reranking precision than Option A.
- **Model size on disk**: ~64MB + ~63MB = **~127MB**

### Option C: Long Context for Code
- **Embedding**: model2vec/potion-code-16M (via model2vec-rs)
- **Reranker**: jinaai/jina-reranker-v1-tiny-en (33M, ONNX, 8K context)
- **Expected latency**: <1ms embed + ~200ms rerank (20 candidates) = **~200ms total**
- **Quality**: Best for long code chunks. 8K context avoids truncation.
- **Model size on disk**: ~64MB + ~130MB = **~194MB**

### Option D: No Reranker (semble approach)
- **Embedding**: model2vec/potion-code-16M
- **Reranker**: None (rely on BM25 + vector hybrid scoring only)
- **Expected latency**: **<2ms total**
- **Quality**: Comparable to semble's 0.854 NDCG@10 when combined with BM25 hybrid.
- **Model size on disk**: **~64MB**

---

## Critical Details for Implementation

### model2vec-rs Integration
- Crate: `model2vec-rs` on crates.io (also `model2vec` on crates.io from a different maintainer)
- Official repo: https://github.com/MinishLab/model2vec-rs
- Supports: safetensors loading, f32/f16/i8 weights, batch processing
- No ONNX runtime dependency. Pure Rust with safetensors + tokenizers.
- The model files are just: `model.safetensors` (embedding matrix) + `tokenizer.json`
- Embedding process: tokenize -> lookup embeddings from matrix -> mean pool -> normalize

### potion-code-16M Specifics
- Created: April 2026
- Trained on: cornstack code datasets (Python, Java, PHP, Go, JS, Ruby)
- Distilled from: nomic-ai/CodeRankEmbed (137M params)
- Output dimension: 256
- Model size: ~64MB (safetensors)
- License: MIT

### ONNX Reranker Integration
- TinyBERT-L2-v2 and MiniLM-L2-v2 both have official ONNX exports on HuggingFace
- Already tagged with `onnx` in their HF repos
- Compatible with Vera's existing ONNX Runtime setup
- INT8 quantization via ONNX can further reduce latency by ~30-50%

### Quantization Options for Rerankers
- ONNX O3 optimization + INT8 quantization gives the best CPU speedup (from sbert benchmarks)
- OpenVINO backend with qint8 is fastest on Intel CPUs specifically
- Standard ONNX with avx512_vnni quantization is good cross-platform

---

## Summary Recommendation

**Start with Option A** (potion-code-16M + TinyBERT-L2-v2). This gives:
1. Near-instant embedding via native Rust (no ONNX for embeddings)
2. Fast reranking via a tiny 4.4M param ONNX model
3. Total query latency well under 100ms on CPU
4. Small total model download (~81MB vs current ~500MB+)
5. Code-specific embeddings trained on the same data as state-of-the-art models

The static embedding + tiny reranker combo is the right architecture for CPU-only use. The embedder handles fast recall (finding candidates) while the reranker provides the precision boost. This matches Vera's existing retrieve-then-rerank pipeline.

If quality proves insufficient, step up to Option B (MiniLM-L2 reranker) for ~2.5x model size but noticeably better ranking. The embedding side should stay as potion-code-16M regardless, since there's no code-specific transformer model small enough to beat it on CPU.
