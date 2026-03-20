// Minimal CLI that just prints a message and exits.
// Used to measure cold-start overhead of a Bun/TS process.
const args = process.argv.slice(2);

if (args[0] === "--version") {
  console.log("spike-startup 0.1.0");
} else {
  console.log(
    JSON.stringify(
      {
        tool: "ts-bun",
        operation: "startup",
        status: "ok",
      },
      null,
      2
    )
  );
}
