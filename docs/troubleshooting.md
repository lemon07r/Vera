# Troubleshooting

## "No index found in current directory"

Either the repository hasn't been indexed yet, or you're running the command from the wrong directory.

```bash
vera index .
```

Make sure you're in the repository root (the directory containing `.vera/`).

## Results feel stale or outdated

Code changed after the last index. Update it:

```bash
vera update .
```

## Local ONNX inference isn't working

Run the diagnostic command first:

```bash
vera doctor
vera doctor --probe
vera doctor --probe --json
vera upgrade
```

Common causes:

- Models haven't been downloaded yet. run `vera setup` (or `vera setup --onnx-jina-cpu`)
- If assets are missing or partially downloaded, run `vera repair` after fixing the underlying environment issue
- ONNX Runtime auto-download failed. check network, or set `ORT_DYLIB_PATH` to a manually installed library
- If your network only allows browser downloads, use [manual-install.md](manual-install.md)
- GPU backend not working. make sure the required drivers are installed (CUDA 12+ for `--onnx-jina-cuda`, ROCm for `--onnx-jina-rocm`, DirectX 12 for `--onnx-jina-directml`). CoreML (`--onnx-jina-coreml`) requires macOS on Apple Silicon. OpenVINO (`--onnx-jina-openvino`) and ROCm (`--onnx-jina-rocm`) are installed automatically via pip; if the automatic install fails, install manually (`pip install onnxruntime-openvino` or `pip install onnxruntime-rocm`), then set `ORT_DYLIB_PATH` to the `libonnxruntime.so` inside the package. If GPU init still fails, rerun with `--onnx-jina-cpu` or fix the provider-specific dependencies.
- `vera doctor` will flag missing models or runtime, show the saved and active backend, print the installed Vera version, and check for newer releases. `vera doctor --probe` adds a deeper read-only session probe and does not download or repair missing assets. `vera repair` is the write path if you need Vera to re-fetch local assets. `vera upgrade` shows the binary update plan and can apply it when the install method is known.

If Vera now fails fast with a message like `CUDA backend selected, but required libraries are missing`, the ONNX Runtime CUDA provider was downloaded but your system linker cannot find the CUDA or cuDNN shared libraries it depends on. Install the required userspace libraries, refresh the linker cache if needed, then rerun `vera doctor --probe`.

## API mode isn't working

Check that all three environment variables are set:

- `EMBEDDING_MODEL_BASE_URL`
- `EMBEDDING_MODEL_ID`
- `EMBEDDING_MODEL_API_KEY`

If you're using a reranker, its three variables (`RERANKER_MODEL_BASE_URL`, `RERANKER_MODEL_ID`, `RERANKER_MODEL_API_KEY`) must either all be set or all be absent. Partial configuration will fail.

Re-run setup to persist a working configuration:

```bash
vera setup --api
```

## GPU runs out of memory during indexing

Vera auto-detects VRAM and adjusts batch size, but very low-VRAM GPUs (4 GB or less) may still run out of memory. Use the `--low-vram` flag:

```bash
vera index . --low-vram
```

This forces batch size 1 and caps the ONNX Runtime memory arena to 1 GB. You can also manually tune batch size with `vera config set embedding.batch_size 1`.

On newer builds, Vera does not send every local GPU batch to ONNX at the configured `embedding.batch_size`. It tokenizes first, shrinks long-sequence micro-batches aggressively, and learns safer limits per sequence-length bucket from real runs. Those learned windows are stored in `~/.vera/adaptive-batch-scaler.json` and reused on later runs for the same backend, device, and model. If a pathological batch still trips an allocation error, Vera retries it at a smaller size instead of aborting the whole index.

If you still see repeated retries or very slow indexing, lower `embedding.batch_size` manually or use `--low-vram`.

## Too many irrelevant results

Try narrowing your search:

- `--lang rust`: filter by language
- `--path "src/**/*.ts"`: filter by file path
- `--type function`: filter by symbol type
- `--limit 5`: return fewer results
- Rewrite the query to be more specific about the behavior you're looking for

See the [query guide](query-guide.md) for more tips on writing effective queries.

## Need an exact text match?

Vera is a semantic search tool. For exact string or regex matching, use `rg` (ripgrep) instead:

```bash
rg "EMBEDDING_MODEL_BASE_URL"
rg "TODO\(" -n
```
