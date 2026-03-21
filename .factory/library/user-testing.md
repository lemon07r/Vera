# User Testing

Testing surface, resource cost classification, and validation approach.

**What belongs here:** How to test Vera's user-facing surfaces, concurrency limits, testing infrastructure.

---

## Validation Surface

**Primary surface:** CLI (terminal commands)
- `vera index <path>` - index a repository
- `vera search <query>` - search indexed code
- `vera update <path>` - incremental re-index
- `vera stats` - project statistics
- `vera config` - configuration management
- All commands support `--json` output mode

**Secondary surface:** MCP server (Milestone 3+)
- Exposed via `vera mcp` command
- Tools: search_code, index_project, update_project, get_stats

**Testing tools:** Execute tool for CLI invocation, output parsing, exit code checking.
No browser testing needed.

## Validation Concurrency

**Surface: CLI**
- Each validator runs CLI commands (~50-200MB RAM per invocation, ~500MB peak during indexing)
- Dev server: none needed (CLI tool)
- Infrastructure shared: test repo clones in `.bench/repos/`
- Max concurrent validators: **5** (19GB available / 0.5GB peak per validator * 0.7 headroom = ~26, capped at 5)
- Isolation: each validator can work on the same indexed repo since search is read-only. Index/update operations need file-level isolation.

**Surface: MCP**
- MCP server runs on port 3200
- Only one MCP server instance at a time
- Max concurrent validators for MCP surface: **1** (single server)

## Test Repositories

Benchmark corpus stored in `.bench/repos/` (gitignored). Repos pinned to specific commit SHAs for reproducibility.
Corpus repos: ripgrep (Rust), flask (Python), fastify (TypeScript), turborepo (Polyglot).
Setup: `bash eval/setup-corpus.sh` or verify with `cargo run --manifest-path eval/Cargo.toml --bin vera-eval -- verify-corpus`.

## Flow Validator Guidance: CLI

**Testing tool:** `Execute` tool for running CLI commands and checking output/exit codes.

**Isolation rules:**
- All eval-foundation testing is read-only (running the eval harness, inspecting files).
- No index/update operations that could cause write contention.
- Multiple validators can safely read `.bench/repos/` and `eval/tasks/` concurrently.
- The eval harness runs with mock adapters (no real tool invocations), so no external API calls needed.

**Boundaries:**
- Each validator should write its evidence files to its assigned evidence directory under `{missionDir}/evidence/eval-foundation/<group-id>/`.
- Each validator should write its flow report to `.factory/validation/eval-foundation/user-testing/flows/<group-id>.json`.

**Key commands:**
- Build eval harness: `cargo build --manifest-path eval/Cargo.toml`
- Run harness: `cargo run --manifest-path eval/Cargo.toml --bin vera-eval -- run`
- Run with JSON only: `cargo run --manifest-path eval/Cargo.toml --bin vera-eval -- run --json-only`
- Verify corpus: `cargo run --manifest-path eval/Cargo.toml --bin vera-eval -- verify-corpus`
- Stability check: `cargo run --manifest-path eval/Cargo.toml --bin vera-eval -- stability`

**ADR location:** `docs/adr/` — all ADRs follow a 7-section format.

## Flow Validator Guidance: File Verification

For architecture decision assertions, validators verify that ADR files exist at `docs/adr/` with the required 7-section format (Question, Options, Evaluation Method, Evidence, Decision, Consequences, Follow-up) and contain concrete evidence data.

## Flow Validator Guidance: Core Engine CLI

**Testing tool:** `Execute` tool for running CLI commands and checking output/exit codes.

**Vera binary:** `/home/lamim/Development/Tools/Vera/target/release/vera`

**API credentials:** Source secrets.env before running commands that need embedding/reranking:
```bash
set -a && source /home/lamim/Development/Tools/Vera/secrets.env && set +a
```

**Index locations:** Vera stores index in `.vera/` directory inside the repo being indexed. To search, `cd` into the indexed repo first, then run `vera search`.

**Pre-indexed repos:**
- Flask (Python): `/home/lamim/Development/Tools/Vera/.bench/repos/flask` — INDEXED
- Fastify (TypeScript): `/home/lamim/Development/Tools/Vera/.bench/repos/fastify` — INDEXED
- Ripgrep (Rust): `/home/lamim/Development/Tools/Vera/.bench/repos/ripgrep` — INDEXED
- Turborepo (Polyglot): `/home/lamim/Development/Tools/Vera/.bench/repos/turborepo` — NOT INDEXED

**Known issues:**
- Reranker API (SiliconFlow) may have connectivity issues. Vera should degrade gracefully (return unreranked results with warning). This is expected behavior for VAL-RET-012.
- Embedding API (SiliconFlow Qwen3) may occasionally timeout. Retry or accept BM25 fallback.

**Isolation rules for core-engine testing:**
- Search operations are read-only and safe to run concurrently.
- Index operations write to `.vera/` in the target repo — use separate repos or temp copies for index tests to avoid write contention.
- For index error handling tests (invalid path, empty dir, binary files, permission errors), use temporary directories to avoid interfering with other validators.
- Each validator writes evidence to its assigned evidence directory.

**Key commands:**
- Build: already built at `/home/lamim/Development/Tools/Vera/target/release/vera`
- Index: `cd <repo-dir> && set -a && source /home/lamim/Development/Tools/Vera/secrets.env && set +a && /home/lamim/Development/Tools/Vera/target/release/vera index .`
- Search: `cd <repo-dir> && set -a && source /home/lamim/Development/Tools/Vera/secrets.env && set +a && /home/lamim/Development/Tools/Vera/target/release/vera search "<query>" --json`
- Stats: `cd <repo-dir> && /home/lamim/Development/Tools/Vera/target/release/vera stats --json`

**Source sizes for reference (files only, excluding .git/.vera/node_modules):**
- Run `find <repo> -not -path '*/.git/*' -not -path '*/.vera/*' -not -path '*/node_modules/*' -type f | xargs du -sb | awk '{sum+=$1} END{print sum}'` to get source size.

## Flow Validator Guidance: Agent Integration

### Common Setup

**Vera binary:** `/home/lamim/Development/Tools/Vera/target/release/vera`

**API credentials:** Source secrets.env before running commands that need embedding/reranking:
```bash
set -a && source /home/lamim/Development/Tools/Vera/secrets.env && set +a
```

**Pre-indexed repos (read-only safe):**
- Flask (Python): `/home/lamim/Development/Tools/Vera/.bench/repos/flask` — INDEXED
- Fastify (TypeScript): `/home/lamim/Development/Tools/Vera/.bench/repos/fastify` — INDEXED
- Ripgrep (Rust): `/home/lamim/Development/Tools/Vera/.bench/repos/ripgrep` — INDEXED

**SKILL.md location:** `/home/lamim/Development/Tools/Vera/SKILL.md`
**Cargo.toml location:** `/home/lamim/Development/Tools/Vera/Cargo.toml`

### Isolation Boundaries

**CLI read-only groups** (CLI completeness, agent capsules): Use pre-indexed repos at `.bench/repos/`. Safe to run concurrently — search and stats are read-only.

**Incremental indexing group**: Use isolated repo copy at `/tmp/vera-test-incr/flask`. This group has exclusive write access. Do NOT use `.bench/repos/` for modification tests.

**Cross-area flows group**: Use isolated repo copy at `/tmp/vera-test-cross/flask`. This group has exclusive write access. Do NOT use `.bench/repos/` for modification tests.

**MCP group**: Use isolated repo copy at `/tmp/vera-test-mcp/flask`. MCP server uses stdio transport (not TCP), so invoke via pipe. Only one MCP instance at a time.

### MCP Testing Approach

The MCP server uses stdio JSON-RPC transport. Test by piping JSON-RPC messages:
```bash
# Initialize and send requests via pipe
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1.0"}}}' | vera mcp
```

For multi-message testing, use a script that sends multiple JSON-RPC messages on stdin and reads responses from stdout. The server processes one request per line.

### Local Mode Validation Notes

- Build the release binary with `cargo build --release --features local` before validating `--local` flows or `VERA_LOCAL=1` behavior.
- Use `VERA_LOCAL=1` when validating MCP local mode, since `vera mcp` reads the environment flag rather than a CLI switch.
- Local-mode assertions can run without API keys, but first-run evidence may include model downloads into `~/.vera/models/`.

### Known Issues for Agent Integration

- MCP server uses stdio transport, NOT HTTP/TCP. No port needed.
- Reranker API may have connectivity issues — Vera degrades gracefully.
- Embedding API may occasionally timeout — accept BM25 fallback for search tests.
- `vera update` requires the target directory to already have a `.vera/` index directory.

## Flow Validator Guidance: Polish-Release Benchmarks

**Testing tool:** `Execute` tool for inspecting benchmark result files and reports.

**All assertions in this group are read-only verification** — no Vera CLI commands needed, no API credentials needed. You are verifying that benchmark artifacts exist and contain required content.

**Key file locations:**
- Benchmark reports: `/home/lamim/Development/Tools/Vera/benchmarks/reports/`
  - `final-benchmark-suite.md` — main benchmark report
  - `reproduction-guide.md` — reproduction instructions
  - `ablation-studies.md` — ablation study results
  - `competitor-baselines.md` — competitor baseline results
- Benchmark results (JSON): `/home/lamim/Development/Tools/Vera/benchmarks/results/`
  - `final-suite/combined_results.json` — combined results
  - `final-suite/vera_hybrid_results.json` — Vera hybrid results
  - `final-suite/vera_bm25_only_results.json` — BM25-only results
  - `ablation-studies/ablation_results.json` — ablation data
- Eval tasks: `/home/lamim/Development/Tools/Vera/eval/tasks/` — task definitions with ground truth

**What to verify for each assertion:**
- VAL-BENCH-001: A single command produces complete benchmark report with all tasks, metrics, baselines, Vera results
- VAL-BENCH-002: Vera outperforms lexical baseline (ripgrep) on semantic/intent tasks by 10%+ relative on Recall@5 or MRR
- VAL-BENCH-003: Vera hybrid Recall@1 exceeds pure vector-only on exact symbol lookup tasks
- VAL-BENCH-004: Ablation studies documented for: hybrid vs semantic-only, hybrid vs lexical-only, reranker on/off, 2+ embedding models, each with per-category breakdown
- VAL-BENCH-005: Performance targets met: 100K LOC index <120s, BM25 p95 <10ms, cached hybrid p95 <100ms, incremental <5s, index <2x source
- VAL-BENCH-006: Formatted comparison table suitable for README, reproduction guide with versions/setup/commands/expected ranges

**Isolation:** All read-only. No contention concerns.

## Flow Validator Guidance: Polish-Release Documentation

**Testing tool:** `Execute` tool for reading files and running CLI verification commands.

**Vera binary:** `/home/lamim/Development/Tools/Vera/target/release/vera`

**Key file locations:**
- README: `/home/lamim/Development/Tools/Vera/README.md`
- SKILL.md: `/home/lamim/Development/Tools/Vera/SKILL.md`
- Cargo.toml: `/home/lamim/Development/Tools/Vera/Cargo.toml`
- ADRs: `/home/lamim/Development/Tools/Vera/docs/adr/` (001-004 + 000-decision-summary)
- Recommendation memo: `/home/lamim/Development/Tools/Vera/docs/recommendation-memo.md`
- Maintainability audit: `/home/lamim/Development/Tools/Vera/docs/maintainability-audit.md`

**What to verify for each assertion:**
- VAL-DOCS-001: README has installation (prerequisites, install cmd, verification), usage quickstart (index + search with sample output), benchmark results table
- VAL-DOCS-002: SKILL.md covers purpose, when-to-use, commands with examples, output interpretation, query tips. Under 2000 tokens (~8000 chars). Every documented command works
- VAL-DOCS-003: All ADRs in Accepted/Superseded status, none Draft/Proposed
- VAL-DOCS-004: Recommendation memo covers: chosen architecture + rationale, rejected paths + reasons, open risks, next steps
- VAL-DOCS-005: Maintainability audit covers: module sizes vs budgets, function sizes vs budgets, ownership boundaries, test coverage, dead code
- VAL-DOCS-006: No dead experimental code in main branch, spikes in archive or labeled

**For VAL-DOCS-002 command verification:** Source secrets.env before running vera commands:
```bash
set -a && source /home/lamim/Development/Tools/Vera/secrets.env && set +a
```
Pre-indexed repos at `.bench/repos/flask`, `.bench/repos/fastify`, `.bench/repos/ripgrep` are available.

**Isolation:** Mostly read-only file inspection. CLI command testing uses pre-indexed repos (read-only search). Safe to run concurrently with other groups.

## Flow Validator Guidance: Repo Cleanup

**Testing tool:** `Execute` tool for repository inspection commands and output/exit-code checks.

**Scope:** Validate the repository-cleanup milestone through the repo's real CLI/user surface: `git`, shell inspection commands, and Cargo validators that a maintainer would run locally.

**Isolation rules:**
- These assertions are read-only against the repository checkout and git index.
- Validators may run concurrently because they should not modify tracked files, git state, or `.dev/` contents.
- If a validator needs temporary scratch files for parsing output, keep them under its assigned evidence directory only.

**Boundaries:**
- Repo root: `/home/lamim/Development/Tools/Vera`
- Mission dir: `/home/lamim/.factory/missions/f17cfc12-1e02-4226-b7e0-40720f4f4993`
- Evidence must stay under the assigned evidence directory for the validator group.

**Key commands:**
- `git -C /home/lamim/Development/Tools/Vera status --short`
- `git -C /home/lamim/Development/Tools/Vera ls-files`
- `git -C /home/lamim/Development/Tools/Vera check-ignore .dev/`
- `ls -la /home/lamim/Development/Tools/Vera`
- `find /home/lamim/Development/Tools/Vera/.dev -type f | wc -l`
- `cargo test`, `cargo build`, `cargo clippy -- -D warnings`, `cargo fmt --check`

**What to verify:**
- `.gitignore` ignores `.factory/`, `.agents/`, and `.dev/`; git status shows none of them as untracked.
- Root AI artifacts are no longer tracked by git.
- `.dev/` exists on disk, is organized, contains archive content, and remains untracked.
- README has no placeholder URL or Factory-process references.
- `secrets.env` is not tracked and tracked files do not expose common secret patterns.
- Repository root public-structure assertions are based on `git ls-files` (tracked top-level files/directories), not a raw `ls` of all on-disk local files; gitignored paths like `secrets.env`, `target/`, `.factory/`, and `.dev/` may still exist locally.

## Flow Validator Guidance: Extended Languages CLI

**Testing tool:** `Execute` tool for running Vera CLI commands against isolated sample repositories.

**Vera binary:** `/home/lamim/Development/Tools/Vera/target/release/vera`

**API credentials:** Source secrets before any `index` or `search` command that needs embeddings:
```bash
set -a && source /home/lamim/Development/Tools/Vera/secrets.env && set +a
```

**Isolation rules:**
- Each validator gets an exclusive temp repo under `/tmp/vera-usertest-extended/<group-id>/repo`.
- Validators may create, edit, and index files only inside their assigned temp repo and evidence directory.
- Do not reuse another validator's `.vera/` index or temp files.
- Search and stats are read-only after indexing within the assigned repo.

**Boundaries:**
- Repo root: `/home/lamim/Development/Tools/Vera`
- Mission dir: `/home/lamim/.factory/missions/f17cfc12-1e02-4226-b7e0-40720f4f4993`
- Evidence dir: `{missionDir}/evidence/extended-languages/<group-id>/`
- Flow report path: `.factory/validation/extended-languages/user-testing/flows/<group-id>.json`

**Suggested verification pattern:**
- Create representative source files for the assigned languages.
- Run `vera index .` inside the temp repo.
- Run `vera search "<symbol>" --json` for each assigned language.
- Verify JSON results include the expected `language`, `symbol_type`, and symbol/file match.
- For documentation assertions, inspect `README.md` and `SKILL.md` and confirm the supported language list is explicit.

**Notes:**
- No long-running services are needed for this milestone.
- Rebuild the release binary with `cargo build --release` before running extended-language CLI validation so `target/release/vera` matches the current source tree.
- Keep evidence concise: command outputs, parsed JSON snippets, and any reasoning tying grouped assertions back to observed CLI behavior.

## Flow Validator Guidance: Local Inference CLI

**Testing tool:** `Execute` tool for release builds, CLI invocations, JSON validation, and metadata inspection.

**Recommended concurrency for this surface:** **2 validators max** for local-inference user testing. Although general CLI validation can scale higher, this milestone adds release builds and ONNX model initialization/downloads that are materially heavier on CPU, memory, disk, and network.

**Shared setup produced by the coordinator:**
- Default-feature release binary: `/tmp/vera-usertest-local-inference/bin/vera-default`
- Local-feature release binary: `/tmp/vera-usertest-local-inference/bin/vera-local`
- Shared temp root: `/tmp/vera-usertest-local-inference/`

**Isolation rules:**
- Each validator gets an exclusive workspace under `/tmp/vera-usertest-local-inference/<group-id>/`.
- Create sample repos only inside the assigned workspace.
- For first-run local-model download assertions, set `HOME=/tmp/vera-usertest-local-inference/<group-id>/home` so `~/.vera/models/` is isolated per validator and download behavior is repeatable.
- Do not reuse another validator's temp repo, HOME directory, or `.vera/` index.
- API-mode assertions may source `/home/lamim/Development/Tools/Vera/secrets.env`; local-mode assertions should intentionally avoid API credentials unless the assertion explicitly needs a provider mismatch comparison.

**Useful command patterns:**
- Build default binary: `CARGO_TARGET_DIR=/tmp/vera-usertest-local-inference/target-default cargo build --release`
- Build local binary: `CARGO_TARGET_DIR=/tmp/vera-usertest-local-inference/target-local cargo build --release --features local`
- API-mode index/search: `set -a && source /home/lamim/Development/Tools/Vera/secrets.env && set +a && /tmp/vera-usertest-local-inference/bin/vera-default ...`
- Local-mode index/search: `HOME=/tmp/vera-usertest-local-inference/<group-id>/home /tmp/vera-usertest-local-inference/bin/vera-local ...`
- Metadata inspection: `python3 - <<'PY' ... sqlite3 connect to .vera/metadata.db ... PY`

**Local-inference assertion tips:**
- `VAL-LOCAL-001` / `VAL-LOCAL-009`: verify the default build with `cargo tree --features ''` / `nm` or `strings` checks against the default binary copy.
- `VAL-LOCAL-005`: keep an isolated HOME with no cached models before the first local invocation; capture stderr showing `Downloading https://huggingface.co/...` and subsequent progress lines, then rerun to confirm no download output.
- `VAL-LOCAL-006` / `VAL-LOCAL-007`: local commands should succeed without sourcing API credentials.
- `VAL-LOCAL-008`: create both API-indexed and local-indexed sample repos, then search/update with the opposite provider and capture the mismatch error text.
- `VAL-LOCAL-010`, `VAL-CROSS-101`, `VAL-CROSS-102`: use a small isolated repo containing at least one newly added language file (for example Ruby and C#) so language classification and symbol metadata can be asserted from `--json` results.
- `VAL-CROSS-103`: run the full validator commands (`cargo test`, `cargo test --features local`, `cargo clippy -- -D warnings`, `cargo clippy --features local -- -D warnings`, `cargo fmt --check`) from the repo root and record exit codes plus notable output.

**Observed local-inference frictions (2026-03-20 / 2026-03-21 validation):**
- On hosts where `/tmp` is nearly full, keep the assigned `HOME` under `/tmp` for isolation but point `HOME/.vera` and `TMPDIR` at evidence-backed storage on the main filesystem; this preserved a cold-cache validation path for `cargo test --features local` and `cargo clippy --features local -- -D warnings` during the 2026-03-21 rerun.
- Faithful local-mode indexing of the copied standard benchmark repos used for the 17-task Recall@5 check still terminated during the 2026-03-21 rerun (`ripgrep` exit `-9`, `flask` exit `-15`, `fastify` exit `-15`) even with the reduced local defaults (`embedding.batch_size=16`, `embedding.max_concurrent_requests=1`). Treat this as a validation failure signal rather than silently downgrading the workload.

## Flow Validator Guidance: Local Inference MCP

**Testing tool:** `Execute` tool for stdio JSON-RPC interaction with the MCP server.

**Recommended concurrency for this surface:** **1 validator max**. Only one local-inference MCP validator should run at a time.

**Isolation rules:**
- Use an exclusive workspace under `/tmp/vera-usertest-local-inference/<group-id>/`.
- Use `HOME=/tmp/vera-usertest-local-inference/<group-id>/home` to isolate model downloads and caches.
- Use `VERA_LOCAL=1` when validating local-mode MCP behavior.
- MCP uses stdio only; do not bind ports.

**Suggested verification pattern:**
- Prepare or reuse an isolated indexed repo inside the validator workspace.
- Send newline-delimited JSON-RPC messages to `/tmp/vera-usertest-local-inference/bin/vera-local mcp` for `initialize`, `tools/list`, and the assigned tool call (for example `search_code`).
- Confirm success responses preserve the same JSON schema as CLI output and surface local-mode results without API credentials.
