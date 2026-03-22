# User Testing

Testing surface, required tools, and resource cost classification.

**What belongs here:** How to validate Vera's user-facing behavior, testing tools, concurrency limits.

---

## Validation Surface

**Primary surface:** CLI commands (shell invocations)

| Command | What it tests |
|---------|--------------|
| `vera --version` | Binary works, version correct |
| `vera --help` | Help output, command listing |
| `vera index .` | API mode indexing |
| `vera index --local .` | Local mode with model download |
| `vera search "query"` | Search in API mode |
| `vera search --local "query"` | Search in local mode |
| `vera search --lang rust "query"` | Language-filtered search |
| `vera update .` | Incremental index update |
| `vera stats` | Index statistics display |
| `vera mcp` | MCP server (JSON-RPC stdio) |

**Tools:** Shell commands via Execute tool. No browser or TUI testing needed.

## Validation Concurrency

**Machine:** 12 cores, 30GB RAM, ~14GB available
**CLI invocations:** Lightweight (~50MB each). Max concurrent validators: **5**.
**Benchmark runs:** CPU-intensive (uses all cores). Max concurrent: **1**.
**Local inference:** ~2GB RAM for models. Max concurrent with local: **3**.

## Testing Notes

- API mode requires `secrets.env` loaded (embedding + reranker endpoints)
- Local mode requires ONNX Runtime shared library on the system
- On this Linux host, `ldconfig -p` exposes only `/opt/intel/oneapi/compiler/2025.0/lib/libonnxruntime.1.12.22.721.so`, which is too old for `ort 2.0.0-rc.11`; local validation must point `ORT_DYLIB_PATH` at a compatible 1.23+ runtime instead.
- A compatible temporary runtime was provisioned at `/tmp/vera-onnxruntime-1.24.4/lib/libonnxruntime.so` for simplify-install user-testing reruns.
- The prebuilt validation binary at `/home/lamim/Development/Tools/Vera/target/release/vera` works with that runtime for isolated local-mode fixtures.
- `vera stats` currently has no `--index-dir` flag; run it from the indexed fixture directory (`cd <fixture> && vera stats --json`).
- Benchmark corpus repos must be in `.bench/repos/` (run `eval/setup-corpus.sh`)
- MCP testing: pipe JSON-RPC messages via stdin, read JSON-RPC responses from stdout

## Flow Validator Guidance: CLI

- Use read-only shell commands and file inspection for documentation validation; do not modify repository files.
- Stay inside `/home/lamim/Development/Tools/Vera` and its mission artifacts when gathering evidence.
- Treat the repository working tree as shared state: no `cargo fmt`, no generated outputs, no staging/commits.
- Prefer direct evidence commands (`test`, `git check-ignore`, `rg`, `wc`, `head`/`sed` alternatives via file reads when available) and capture exact outputs in the flow report.
- Validation for docs-restructure is independent and low-cost, so a single CLI validator can cover all assigned assertions serially.
- For `simplify-install`, treat `target/release/vera` as a shared read-only artifact after the build validator finishes; do not rebuild concurrently with other validators.
- For local-mode assertions, use isolated temporary `HOME` directories so model-cache state is deterministic and does not mutate the user's real `~/.vera/models/`.
- Source `secrets.env` with shell environment export (`set -a; . ./secrets.env; set +a`) without echoing variable values when API-mode validation is required.
- To validate the graceful ONNX Runtime failure path, prefer an isolated command with `ORT_DYLIB_PATH` pointed at a nonexistent file instead of altering system libraries.
