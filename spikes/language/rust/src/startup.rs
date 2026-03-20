use std::env;

fn main() {
    // Minimal CLI that just prints a message and exits.
    // Used to measure cold-start overhead of a Rust binary.
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "--version" {
        println!("spike-startup 0.1.0");
    } else {
        println!("{{");
        println!("  \"tool\": \"rust\",");
        println!("  \"operation\": \"startup\",");
        println!("  \"status\": \"ok\"");
        println!("}}");
    }
}
