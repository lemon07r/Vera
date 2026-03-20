# Environment

Environment variables, external dependencies, and setup notes.

**What belongs here:** Required env vars, external API keys/services, dependency quirks, platform-specific notes.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

---

## API Credentials

Stored in `secrets.env` at repo root (NEVER committed, in .gitignore).

- `EMBEDDING_MODEL_BASE_URL` - Qwen3 embedding API endpoint
- `EMBEDDING_MODEL_ID` - Qwen3 embedding model identifier
- `EMBEDDING_MODEL_API_KEY` - API key for embedding service
- `RERANKER_MODEL_BASE_URL` - Qwen3 reranker API endpoint
- `RERANKER_MODEL_ID` - Qwen3 reranker model identifier
- `RERANKER_MODEL_API_KEY` - API key for reranker service

Workers must `source secrets.env` before running tests that need embedding/reranking.
NEVER log, print, or commit API keys.

### Alternative API Providers

- **SiliconFlow** (`api.siliconflow.com/v1`): Used for Qwen3-Embedding-0.6B (lightweight fallback model). Authenticated with the `RERANKER_MODEL_API_KEY` env var (key reuse — same provider). Discovered during embedding-and-chunking-spike.

## Spike Development Notes

- Spike Cargo.toml files in subdirectories (e.g., `spikes/storage/Cargo.toml`) need an empty `[workspace]` table to avoid being pulled into the root workspace. Without this, `cargo build` fails with "current package believes it's in a workspace".
- Spike `Cargo.lock` files should be gitignored (add to spike `.gitignore`). The root `Cargo.lock` IS committed (correct for the main binary project).
- Spike build artifacts (`target/`, `node_modules/`) should be excluded via a local `.gitignore` in the spike directory.

## Nebius API Quirks

- **Transient HTTP 400 "Unable to process"**: Under high concurrency, the Nebius embedding API (Qwen3-Embedding-8B) returns HTTP 400 with body containing "Unable to process". This must be treated as a transient rate-limit error and retried with exponential backoff, not as a permanent client error. The code handles this in `embedding/provider.rs`.
- **Rate limits at 8+ concurrent requests**: With `max_concurrent_requests=8` and `batch_size=64`, occasional rate limit (429) errors occur. The pipeline uses exponential backoff with jitter (up to 3 retries).
- **Auth errors (401) should NOT be retried** — they indicate a permanent credential problem.

## Machine Specs

- AMD Ryzen 5 7600X3D (12 threads)
- 30GB RAM (~19GB available)
- 500GB disk free
- Arch Linux

## Toolchain

- Rust 1.94 (cargo, rustc)
- Node 25.8 (for competitor benchmarking)
- Go 1.25 (for grepai/Zoekt benchmarking)
- Python 3.14 (for cocoindex-code benchmarking)
- Docker 29.3
