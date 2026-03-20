use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: spike-walk <directory> [iterations]");
        std::process::exit(1);
    }
    let dir_path = &args[1];
    let iterations: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);

    // Warmup walk
    let mut warmup_count = 0u64;
    for entry in ignore::WalkBuilder::new(dir_path)
        .hidden(false)
        .git_ignore(true)
        .build()
    {
        if let Ok(e) = entry {
            if e.file_type().is_some_and(|ft| ft.is_file()) {
                warmup_count += 1;
            }
        }
    }

    // Timed iterations
    let start = Instant::now();
    let mut total_files = 0u64;
    let mut total_bytes = 0u64;
    for _ in 0..iterations {
        let mut file_count = 0u64;
        let mut byte_count = 0u64;
        for entry in ignore::WalkBuilder::new(dir_path)
            .hidden(false)
            .git_ignore(true)
            .build()
        {
            if let Ok(e) = entry {
                if e.file_type().is_some_and(|ft| ft.is_file()) {
                    file_count += 1;
                    if let Ok(meta) = e.metadata() {
                        byte_count += meta.len();
                    }
                }
            }
        }
        total_files += file_count;
        total_bytes += byte_count;
    }
    let elapsed = start.elapsed();

    let avg_files = total_files / iterations as u64;
    let avg_bytes = total_bytes / iterations as u64;
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let total_ms = elapsed.as_secs_f64() * 1000.0;

    println!("{{");
    println!("  \"tool\": \"rust\",");
    println!("  \"operation\": \"file_tree_walk\",");
    println!("  \"directory\": \"{dir_path}\",");
    println!("  \"warmup_files\": {warmup_count},");
    println!("  \"iterations\": {iterations},");
    println!("  \"total_ms\": {total_ms:.2},");
    println!("  \"avg_ms\": {avg_ms:.3},");
    println!("  \"avg_files\": {avg_files},");
    println!("  \"avg_bytes\": {avg_bytes}");
    println!("}}");
}
