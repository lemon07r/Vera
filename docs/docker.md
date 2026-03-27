# Docker

Vera provides Docker images for running the MCP server (or any Vera command) in a container. Three variants are available:

| Image | Base | Size | Use case |
|-------|------|------|----------|
| `vera:cpu` | `debian:trixie-slim` | ~150 MB | Default, works everywhere |
| `vera:cuda` | `nvidia/cuda:13.1.1-runtime` | ~4 GB | NVIDIA GPU acceleration |
| `vera:rocm` | `rocm/dev-ubuntu-24.04:6.4.4` | ~8 GB | AMD GPU acceleration |

## Running the MCP server

The primary use case is running Vera's MCP server as a background process for your editor or AI agent.

**CPU:**

```bash
docker run --rm -i -v $(pwd):/workspace ghcr.io/lemon07r/vera:cpu
```

**CUDA (NVIDIA):**

```bash
docker run --rm --gpus all -i -v $(pwd):/workspace ghcr.io/lemon07r/vera:cuda
```

**ROCm (AMD):**

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri -i -v $(pwd):/workspace ghcr.io/lemon07r/vera:rocm
```

The container starts `vera mcp` by default (JSON-RPC over stdio). The `-i` flag keeps stdin open for communication. The volume mount gives Vera access to your project files.

## MCP client configuration

Point your MCP client (Claude Desktop, Cursor, etc.) at the Docker container:

```json
{
  "mcpServers": {
    "vera": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "-v", "/path/to/project:/workspace", "ghcr.io/lemon07r/vera:cpu"]
    }
  }
}
```

For GPU variants, add the appropriate device flags to `args`:

```json
{
  "args": ["run", "--rm", "--gpus", "all", "-i", "-v", "/path/to/project:/workspace", "ghcr.io/lemon07r/vera:cuda"]
}
```

## First run

On first run inside the container, Vera will download ONNX models and the ONNX Runtime library to `/root/.vera/`. To persist these across container restarts, mount a volume:

```bash
docker run --rm -i \
  -v $(pwd):/workspace \
  -v vera-models:/root/.vera \
  ghcr.io/lemon07r/vera:cpu
```

After the first run, subsequent starts skip the download.

## Running other commands

Override the default `mcp` command to run any Vera command:

```bash
docker run --rm -v $(pwd):/workspace ghcr.io/lemon07r/vera:cpu index /workspace
docker run --rm -v $(pwd):/workspace ghcr.io/lemon07r/vera:cpu search "authentication logic"
docker run --rm -v $(pwd):/workspace ghcr.io/lemon07r/vera:cpu stats
```

## Building locally

From the repo root:

```bash
docker build -f docker/Dockerfile.cpu -t vera:cpu .
docker build -f docker/Dockerfile.cuda -t vera:cuda .
docker build -f docker/Dockerfile.rocm -t vera:rocm .
```

## GPU requirements

**CUDA:** Requires NVIDIA drivers with CUDA 12+ support and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

**ROCm:** Requires AMD GPU with ROCm 6.x drivers. The `/dev/kfd` and `/dev/dri` devices must be accessible.
