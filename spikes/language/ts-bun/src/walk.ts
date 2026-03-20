import { readdirSync, statSync } from "fs";
import { join } from "path";

const dirPath = process.argv[2];
const iterations = parseInt(process.argv[3] || "5", 10);

if (!dirPath) {
  console.error("Usage: bun run src/walk.ts <directory> [iterations]");
  process.exit(1);
}

// Gitignore-aware walk using Bun's Glob
// Bun has a built-in Glob API: new Bun.Glob(pattern).scan(options)

async function walkWithBunGlob(dir: string): Promise<{ files: number; bytes: number }> {
  let fileCount = 0;
  let byteCount = 0;
  const glob = new Bun.Glob("**/*");
  for await (const path of glob.scan({ cwd: dir, dot: false, onlyFiles: true })) {
    fileCount++;
    try {
      const st = statSync(join(dir, path));
      byteCount += st.size;
    } catch {
      // skip inaccessible files
    }
  }
  return { files: fileCount, bytes: byteCount };
}

// Warmup walk
const warmup = await walkWithBunGlob(dirPath);

// Timed iterations
const start = performance.now();
let totalFiles = 0;
let totalBytes = 0;

for (let i = 0; i < iterations; i++) {
  const result = await walkWithBunGlob(dirPath);
  totalFiles += result.files;
  totalBytes += result.bytes;
}

const elapsed = performance.now() - start;
const avgFiles = Math.round(totalFiles / iterations);
const avgBytes = Math.round(totalBytes / iterations);
const avgMs = elapsed / iterations;

console.log(
  JSON.stringify(
    {
      tool: "ts-bun",
      operation: "file_tree_walk",
      directory: dirPath,
      warmup_files: warmup.files,
      iterations,
      total_ms: parseFloat(elapsed.toFixed(2)),
      avg_ms: parseFloat(avgMs.toFixed(3)),
      avg_files: avgFiles,
      avg_bytes: avgBytes,
    },
    null,
    2
  )
);
