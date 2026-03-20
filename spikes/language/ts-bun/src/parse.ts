import { readFileSync } from "fs";
import { basename, extname } from "path";

// Use Bun's built-in tree-sitter support if available, otherwise fall back to node-tree-sitter
// Note: Bun does not have built-in tree-sitter. We use the node binding.

const Parser = (await import("tree-sitter")).default;
const LangRust = (await import("tree-sitter-rust")).default;
const LangJS = (await import("tree-sitter-javascript")).default;
const LangPython = (await import("tree-sitter-python")).default;

const filePath = process.argv[2];
const iterations = parseInt(process.argv[3] || "10", 10);

if (!filePath) {
  console.error("Usage: bun run src/parse.ts <file-path> [iterations]");
  process.exit(1);
}

const source = readFileSync(filePath, "utf-8");
const lineCount = source.split("\n").length;

const ext = extname(filePath).slice(1);
const parser = new Parser();

const langMap: Record<string, any> = {
  rs: LangRust,
  js: LangJS,
  ts: LangJS, // JS parser handles TS syntax partially
  jsx: LangJS,
  tsx: LangJS,
  py: LangPython,
};

const lang = langMap[ext];
if (!lang) {
  console.error(`Unsupported extension: ${ext}`);
  process.exit(1);
}
parser.setLanguage(lang);

// Warmup parse
parser.parse(source);

// Timed iterations
const start = performance.now();
let nodeCount = 0;

for (let i = 0; i < iterations; i++) {
  const tree = parser.parse(source);
  // Walk all nodes to ensure full parse
  const cursor = tree.walk();
  let reachedRoot = false;
  while (!reachedRoot) {
    nodeCount++;
    if (cursor.gotoFirstChild()) continue;
    if (cursor.gotoNextSibling()) continue;
    let retracing = true;
    while (retracing) {
      if (!cursor.gotoParent()) {
        retracing = false;
        reachedRoot = true;
      } else if (cursor.gotoNextSibling()) {
        retracing = false;
      }
    }
  }
}

const elapsed = performance.now() - start;
const avgMs = elapsed / iterations;

console.log(
  JSON.stringify(
    {
      tool: "ts-bun",
      operation: "tree_sitter_parse",
      file: filePath,
      lines: lineCount,
      bytes: Buffer.byteLength(source, "utf-8"),
      iterations,
      total_ms: parseFloat(elapsed.toFixed(2)),
      avg_ms: parseFloat(avgMs.toFixed(3)),
      nodes_per_iter: Math.round(nodeCount / iterations),
    },
    null,
    2
  )
);
