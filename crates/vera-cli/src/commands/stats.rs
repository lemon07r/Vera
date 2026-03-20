//! `vera stats` — Show index statistics.

use std::process;

/// Run the `vera stats` command.
pub fn run(json_output: bool) -> anyhow::Result<()> {
    let cwd = std::env::current_dir()
        .map_err(|e| anyhow::anyhow!("failed to get current directory: {e}"))?;

    let stats = match vera_core::stats::collect_stats(&cwd) {
        Ok(s) => s,
        Err(err) => {
            eprintln!("Error: {err:#}");
            process::exit(1);
        }
    };

    if json_output {
        let json = serde_json::to_string_pretty(&stats)
            .map_err(|e| anyhow::anyhow!("failed to serialize stats: {e}"))?;
        println!("{json}");
    } else {
        print_human_stats(&stats);
    }

    Ok(())
}

/// Print human-readable index statistics.
fn print_human_stats(stats: &vera_core::stats::IndexStats) {
    println!("Index Statistics");
    println!();
    println!("  Files indexed:   {}", stats.file_count);
    println!("  Total chunks:    {}", stats.chunk_count);
    println!("  Index size:      {}", stats.index_size_human);
    println!();
    if !stats.languages.is_empty() {
        println!("  Language Breakdown:");
        for lang in &stats.languages {
            println!(
                "    {:<15} {:>5} chunks ({:.1}%)",
                lang.language, lang.chunk_count, lang.percentage
            );
        }
    } else {
        println!("  No languages indexed.");
    }
}
