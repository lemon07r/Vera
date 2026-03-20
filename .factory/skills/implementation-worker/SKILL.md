---
name: implementation-worker
description: |
  Implements Rust production code for the Vera CLI tool. Handles tree-sitter parsing,
  AST chunking, indexing, retrieval pipeline, CLI commands, MCP server, incremental indexing,
  language support expansion, local inference integration, and repo cleanup tasks.
  Use when the task requires writing or modifying Rust source code, adding CLI commands,
  building pipeline components, or reorganizing the repository.
---

# Implementation Worker

Implement production Rust code for Vera's core engine, CLI, and agent integration features.
Follow test-driven development: write failing tests first, then implement, then verify.

## When to Use

- The task involves writing new Rust modules or functions
- The task involves adding or modifying CLI commands (`vera index`, `vera search`, etc.)
- The task involves building pipeline stages (parsing, chunking, indexing, retrieval, reranking)
- The task involves the MCP server or JSON output format
- The task involves incremental indexing or file-change detection
- The task is tagged for Milestone 2 or Milestone 3

Do **not** use for research spikes or ADRs (use `research-worker`)
or standalone benchmark runs (use `benchmark-worker`).

## Work Procedure

### Step 1: Understand the Feature Scope

1. Read the feature description. Identify inputs, outputs, and success criteria.
2. Read the relevant ADRs in `docs/adr/` for architecture decisions that constrain this feature.
3. Identify which module(s) this feature belongs to. Check existing module structure:
   - Discovery, parsing, chunking, indexing, retrieval, reranking, output assembly,
     config, CLI interface, MCP interface — each should have clear ownership.
4. Check if there are related existing tests or benchmarks to understand expected behavior.
5. If API credentials are needed (embedding, reranking), confirm `secrets.env` is available.

### Step 2: Write Failing Tests First

1. Create or extend test files in the appropriate module's test directory.
2. Write tests that describe the expected behavior:
   - Unit tests for the core logic (pure functions, data transformations).
   - Integration tests for cross-module flows (index → query → results).
   - Contract tests for any JSON output schemas.
3. Run `cargo test` and confirm the new tests **fail** (they should — the feature doesn't exist yet).
4. If tests need test fixtures (sample source files, expected outputs), add them to `tests/fixtures/`.

### Step 3: Implement the Feature

1. Write the Rust code following these conventions:
   - **Error handling:** Use `thiserror` for library errors, `anyhow` for application-level errors.
     Return `Result<T, E>` — no `.unwrap()` in production code paths.
   - **Types:** Use structured types and enums. Avoid stringly-typed data.
     Derive `Debug`, `Clone`, `Serialize`/`Deserialize` where appropriate.
   - **Modules:** One responsibility per module. If a new module is needed, add it explicitly.
   - **Dependencies:** Use only crates already in `Cargo.toml`. If a new crate is needed,
     justify it and add it via `cargo add <crate>`.
   - **Tree-sitter:** Use the `tree-sitter` Rust crate for parsing. Load grammars via
     the appropriate `tree-sitter-<lang>` crates.

2. Respect size budgets from `vera_agent_guardrails.md`:
   - Files: **< 300 lines** (soft), **< 500 lines** (hard — add a comment explaining why).
   - Functions: **< 40 lines** (soft), **< 80 lines** (hard — add a comment explaining why).
   - If hitting limits, split by responsibility, not arbitrary boundaries.

3. Follow output conventions:
   - **Data goes to stdout.** JSON output must be clean, parseable, and schema-stable.
   - **Logs and diagnostics go to stderr.** Use structured logging (e.g., `tracing` crate).
   - Never print API keys, secrets, or credential-derived data to any output stream.

4. If the feature needs API credentials (embedding/reranking endpoints):
   - Read credentials from environment variables (set via `source secrets.env`).
   - Never hardcode URLs, keys, or model IDs.
   - Never log credential values, even at debug level.

### Step 4: Make Tests Pass

1. Run `cargo test` and iterate until all tests pass (new and existing).
2. Fix any regressions in existing tests before proceeding.
3. If a test is flaky, investigate and fix the root cause — do not skip or ignore it.

### Step 5: Code Quality Checks

1. Run `cargo clippy -- -D warnings` and fix all warnings.
   - Clippy warnings are treated as errors. Do not suppress without justification.
2. Run `cargo fmt --check` and fix any formatting issues.
   - If the project has a `rustfmt.toml`, respect its settings.
3. Review your changes against the size budgets:
   - `wc -l` on modified files. Flag any over 300 lines.
   - Check function lengths in your diff.

### Step 6: Manual Verification

1. Build the binary: `cargo build --release` (or `cargo build` for faster iteration).
2. Run the relevant CLI commands against a real test repository:
   - `vera index <path>` — verify it completes without errors.
   - `vera search "<query>"` — verify results are relevant and JSON is valid.
   - `vera stats` — verify output makes sense.
3. If the feature modifies JSON output, validate the schema:
   - Pipe output through `jq .` to confirm it's valid JSON.
   - Check that required fields are present and correctly typed.
4. If the feature involves the MCP server:
   - Start the server and test with a sample MCP client or `curl`.
   - Verify tool responses match expected format.

### Step 7: Commit

1. Stage changes and review the diff: `git diff --cached`.
2. Write a meaningful commit message:
   - Format: `feat(<module>): <what>` or `fix(<module>): <what>`
   - Example: `feat(retrieval): implement BM25+vector hybrid fusion with RRF`
   - Include a brief body if the change is non-obvious.
3. Verify `secrets.env` is not staged: `git diff --cached --name-only | grep secrets`.
4. Commit. Run `cargo test` one final time after committing to confirm nothing broke.

## Example Handoff

```json
{
  "feature": "BM25 full-text indexing over chunk content and metadata",
  "status": "complete",
  "work_done": [
    "Read ADR-003 (storage backend) and ADR-005 (chunking approach) for context",
    "Wrote 8 unit tests for BM25 indexing: term frequency, inverse doc frequency, scoring, multi-field weighting",
    "Wrote 2 integration tests: index a small repo then query and verify ranked results",
    "Confirmed all 10 tests fail before implementation",
    "Implemented bm25 module: Tokenizer, InvertedIndex, BM25Scorer structs",
    "Added metadata field boosting (symbol names 2x, doc comments 1.5x, body 1x)",
    "All 10 new tests pass, all 47 existing tests still pass",
    "cargo clippy clean, cargo fmt clean",
    "Built release binary, indexed flask repo, searched 'request routing' — top 3 results all relevant",
    "Verified JSON output with jq, schema matches contract",
    "bm25.rs is 248 lines, longest function is 35 lines"
  ],
  "files_modified": [
    "src/indexing/mod.rs (added bm25 module declaration)",
    "src/indexing/bm25.rs (new, 248 lines)",
    "tests/indexing/bm25_test.rs (new, 180 lines)",
    "tests/integration/index_query_test.rs (added 2 BM25 test cases)"
  ],
  "test_results": {
    "total": 57,
    "passed": 57,
    "failed": 0,
    "clippy_warnings": 0,
    "fmt_clean": true
  },
  "commits": [
    "b7c8d9e feat(indexing): implement BM25 full-text indexing with metadata boosting"
  ],
  "manual_verification": "Indexed flask (28k LOC) in 4.2s. 'request routing' returns Flask.route decorator, app.add_url_rule, and Blueprint.route as top 3. JSON output valid.",
  "open_questions": [
    "BM25 k1/b parameters are defaults (1.2/0.75) — may need tuning after hybrid integration"
  ]
}
```

## When to Return to Orchestrator

Return when:

- All new tests pass and no existing tests regressed
- `cargo clippy` and `cargo fmt` are clean
- The binary builds and the feature works in manual CLI testing
- Code respects size budgets (or overages are explained)
- Changes are committed with a meaningful message
- `secrets.env` is confirmed not committed

Escalate back early if:

- The feature scope is significantly larger than described
- An architecture decision (ADR) is missing and the feature requires one
- A dependency is needed that isn't in `Cargo.toml` and the choice isn't obvious
- Existing tests break in ways that suggest a design conflict
- The feature requires changes across many modules (possible design issue)
