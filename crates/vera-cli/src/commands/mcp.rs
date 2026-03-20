//! `vera mcp` — Start the MCP server over stdio.

use std::io;

/// Run the MCP server, reading from stdin and writing to stdout.
pub fn run() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = io::BufReader::new(stdin.lock());
    let mut writer = io::BufWriter::new(stdout.lock());

    vera_mcp::server::run_server(&mut reader, &mut writer);
}
