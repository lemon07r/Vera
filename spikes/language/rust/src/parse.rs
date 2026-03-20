use std::env;
use std::fs;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: spike-parse <file-path> [iterations]");
        std::process::exit(1);
    }
    let file_path = &args[1];
    let iterations: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);

    let source = fs::read_to_string(file_path).expect("Failed to read file");
    let line_count = source.lines().count();

    // Detect language from extension
    let ext = file_path.rsplit('.').next().unwrap_or("");
    let mut parser = tree_sitter::Parser::new();
    match ext {
        "rs" => parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .expect("Failed to set Rust language"),
        "js" | "ts" | "jsx" | "tsx" => parser
            .set_language(&tree_sitter_javascript::LANGUAGE.into())
            .expect("Failed to set JS language"),
        "py" => parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .expect("Failed to set Python language"),
        _ => {
            eprintln!("Unsupported extension: {ext}");
            std::process::exit(1);
        }
    }

    // Warmup parse
    let _ = parser.parse(&source, None).expect("Parse failed");

    // Timed iterations
    let start = Instant::now();
    let mut node_count = 0u64;
    for _ in 0..iterations {
        let tree = parser.parse(&source, None).expect("Parse failed");
        // Walk all nodes to ensure full parse
        let mut cursor = tree.walk();
        let mut stack = vec![true]; // true = first visit
        loop {
            if stack.last().copied() == Some(true) {
                node_count += 1;
                *stack.last_mut().unwrap() = false;
                if cursor.goto_first_child() {
                    stack.push(true);
                    continue;
                }
            }
            if cursor.goto_next_sibling() {
                *stack.last_mut().unwrap() = true;
                continue;
            }
            stack.pop();
            if !cursor.goto_parent() {
                break;
            }
        }
    }
    let elapsed = start.elapsed();

    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let total_ms = elapsed.as_secs_f64() * 1000.0;

    println!("{{");
    println!("  \"tool\": \"rust\",");
    println!("  \"operation\": \"tree_sitter_parse\",");
    println!("  \"file\": \"{file_path}\",");
    println!("  \"lines\": {line_count},");
    println!("  \"bytes\": {},", source.len());
    println!("  \"iterations\": {iterations},");
    println!("  \"total_ms\": {total_ms:.2},");
    println!("  \"avg_ms\": {avg_ms:.3},");
    println!("  \"nodes_per_iter\": {}", node_count / iterations as u64);
    println!("}}");
}
