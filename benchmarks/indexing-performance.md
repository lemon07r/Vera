# Indexing Performance

This note captures one representative indexing run on `ripgrep`.

## Environment

- CPU: AMD Ryzen 5 7600X3D
- RAM: 32 GB
- OS: Arch Linux
- Build: release
- Embedding setup: API-backed embeddings with 1024 stored dimensions

## Repository

| Metric | Value |
|--------|-------|
| Repository | `BurntSushi/ripgrep` |
| Commit | `4519153` |
| Total LOC | `175,424` |
| Files parsed | `209` |
| Chunks created | `5,377` |
| Binary files skipped | `9` |
| Wall time | `59.2 s` |
| Source size | `23.4 MB` |
| Index size | `32.4 MB` |
| Size ratio | `1.38x` |

## Storage Breakdown

| Component | Size |
|-----------|------|
| BM25 index | `2.7 MB` |
| Metadata DB | `5.0 MB` |
| Vector DB | `24.7 MB` |

## Notes

- Most of the wall-clock time comes from embedding requests, not parsing or local storage work.
- Parallel parsing and batched writes keep the local portion of indexing short.
- The stored vector dimension is truncated for space efficiency; that reduces index size substantially without changing the public CLI.
