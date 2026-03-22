# Environment

Environment variables, external dependencies, and setup notes.

**What belongs here:** Required env vars, external API keys/services, dependency quirks, platform-specific notes.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

---

## API Credentials

Loaded from `secrets.env` (gitignored, never commit):
- `EMBEDDING_MODEL_BASE_URL` — OpenAI-compatible embedding endpoint
- `EMBEDDING_MODEL_ID` — e.g., `Qwen/Qwen3-Embedding-8B`
- `EMBEDDING_MODEL_API_KEY`
- `RERANKER_MODEL_BASE_URL` — OpenAI-compatible reranker endpoint
- `RERANKER_MODEL_ID` — e.g., `Qwen/Qwen3-Reranker`
- `RERANKER_MODEL_API_KEY`

## Local Inference

- Models stored in `~/.vera/models/` (global, reused across projects)
- Downloads quantized ONNX only (`model_quantized.onnx`)
- Embedding: jina-embeddings-v5-text-nano-retrieval (239M params)
- Reranking: jina-reranker-v2-base-multilingual (278M params)
- Activated by `--local` flag or `VERA_LOCAL=1` env var
- Requires an ONNX Runtime shared library at runtime; if auto-detection fails, set `ORT_DYLIB_PATH` to the library path
- On this host, the linker cache only exposes an incompatible Intel oneAPI ONNX Runtime 1.12 library at `/opt/intel/oneapi/compiler/2025.0/lib/libonnxruntime.1.12.22.721.so`; `ort 2.0.0-rc.11` needs 1.23.x or newer.
- For simplify-install validation, a compatible temporary runtime was provisioned at `/tmp/vera-onnxruntime-1.24.4/lib/libonnxruntime.so` and works when exported via `ORT_DYLIB_PATH`.
- API mode does not need ONNX Runtime installed

## Build Requirements

- Rust 1.85+ (project uses edition 2024)
- C compiler for tree-sitter grammars (cc crate) and bundled SQLite
- On a clean checkout, run `.factory/init.sh` before the first build if vendored grammar sources are missing; it downloads the ignored C grammar directories under `crates/tree-sitter-vue/`, `crates/tree-sitter-dockerfile/`, and `crates/tree-sitter-astro/`
- API mode has no other external runtime dependencies
