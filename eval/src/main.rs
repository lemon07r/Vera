//! Vera Evaluation Harness
//!
//! Single-command benchmark runner that produces structured JSON results
//! alongside human-readable summaries.
//!
//! Usage:
//!   vera-eval run [--tasks-dir <path>] [--output <path>] [--tool <name>]
//!   vera-eval verify-corpus [--corpus <path>]

mod loader;
mod metrics;
mod output;
mod runner;
mod types;
mod vera_adapter;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "vera-eval", about = "Vera evaluation harness")]
#[command(version = env!("CARGO_PKG_VERSION"))]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the benchmark suite and produce evaluation report.
    Run {
        /// Path to the tasks directory (default: eval/tasks/).
        #[arg(long, default_value = "eval/tasks")]
        tasks_dir: PathBuf,

        /// Path to the corpus manifest (default: eval/corpus.toml).
        #[arg(long, default_value = "eval/corpus.toml")]
        corpus: PathBuf,

        /// Output file path for JSON report (default: stdout).
        #[arg(long, short)]
        output: Option<PathBuf>,

        /// Tool adapter to use. `vera-bm25` is the real regression lane; mock tools are for harness self-tests.
        #[arg(long, default_value = "vera-bm25")]
        tool: String,

        /// Suppress human-readable summary (JSON only).
        #[arg(long)]
        json_only: bool,
    },
    /// Verify that corpus repos are cloned at correct SHAs.
    VerifyCorpus {
        /// Path to the corpus manifest.
        #[arg(long, default_value = "eval/corpus.toml")]
        corpus: PathBuf,
    },
    /// Run the harness twice and check result stability.
    Stability {
        /// Path to the tasks directory.
        #[arg(long, default_value = "eval/tasks")]
        tasks_dir: PathBuf,

        /// Path to the corpus manifest (used by real adapters).
        #[arg(long, default_value = "eval/corpus.toml")]
        corpus: PathBuf,

        /// Tool adapter to use. `vera-bm25` is the real stability lane; mock tools are for harness self-tests.
        #[arg(long, default_value = "vera-bm25")]
        tool: String,

        /// Maximum allowed relative difference for retrieval metrics (default: 0.02 = 2%).
        #[arg(long, default_value = "0.02")]
        tolerance: f64,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            tasks_dir,
            corpus,
            output,
            tool,
            json_only,
        } => cmd_run(&tasks_dir, &corpus, output.as_deref(), &tool, json_only),
        Commands::VerifyCorpus { corpus } => cmd_verify_corpus(&corpus),
        Commands::Stability {
            tasks_dir,
            corpus,
            tool,
            tolerance,
        } => cmd_stability(&tasks_dir, &corpus, &tool, tolerance),
    }
}

fn cmd_run(
    tasks_dir: &Path,
    corpus_path: &Path,
    output_path: Option<&Path>,
    tool_name: &str,
    json_only: bool,
) -> Result<()> {
    // Load tasks
    let tasks = loader::load_tasks(tasks_dir)
        .with_context(|| format!("Failed to load tasks from {}", tasks_dir.display()))?;

    if tasks.is_empty() {
        anyhow::bail!("No benchmark tasks found in {}", tasks_dir.display());
    }

    eprintln!("Loaded {} benchmark tasks", tasks.len());

    let report = run_report(&tasks, corpus_path, tool_name)?;

    // Output JSON
    if let Some(path) = output_path {
        output::write_json_report(&report, path)?;
        eprintln!("JSON report written to {}", path.display());
    } else if json_only {
        let json = output::report_to_json(&report)?;
        println!("{json}");
    }

    // Print human-readable summary
    if !json_only {
        output::print_summary(&report, &mut std::io::stderr())?;
    }

    Ok(())
}

fn cmd_verify_corpus(corpus_path: &Path) -> Result<()> {
    let manifest = loader::load_corpus(corpus_path)?;
    let repo_root = std::env::current_dir()?;
    let issues = loader::verify_corpus(&manifest, &repo_root)?;

    if issues.is_empty() {
        println!(
            "✓ All {} repos verified at correct SHAs",
            manifest.repos.len()
        );
        for repo in &manifest.repos {
            println!(
                "  {} ({}) → {}",
                repo.name,
                repo.language,
                &repo.commit[..12]
            );
        }
        Ok(())
    } else {
        eprintln!("✗ Corpus verification failed:");
        for issue in &issues {
            eprintln!("  - {issue}");
        }
        eprintln!("\nRun eval/setup-corpus.sh to fix.");
        std::process::exit(1);
    }
}

fn load_verified_corpus(
    corpus_path: &Path,
) -> Result<(HashMap<String, String>, HashMap<String, String>)> {
    if !corpus_path.exists() {
        anyhow::bail!("Corpus manifest not found at {}", corpus_path.display());
    }

    let manifest = loader::load_corpus(corpus_path)?;
    let repo_root = std::env::current_dir()?;
    let issues = loader::verify_corpus(&manifest, &repo_root)?;
    if !issues.is_empty() {
        anyhow::bail!(
            "Corpus verification failed:\n{}",
            issues
                .into_iter()
                .map(|issue| format!("  - {issue}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    let repo_paths = vera_adapter::repo_paths_from_manifest(&repo_root, &manifest);
    let repo_shas = manifest
        .repos
        .iter()
        .map(|repo| (repo.name.clone(), repo.commit.clone()))
        .collect();

    Ok((repo_paths, repo_shas))
}

fn ensure_task_repos_known(
    tasks: &[types::BenchmarkTask],
    repo_paths: &HashMap<String, String>,
) -> Result<()> {
    let mut missing = tasks
        .iter()
        .filter(|task| !repo_paths.contains_key(&task.repo))
        .map(|task| task.repo.clone())
        .collect::<Vec<_>>();
    missing.sort();
    missing.dedup();

    if missing.is_empty() {
        Ok(())
    } else {
        anyhow::bail!(
            "Tasks reference repos missing from the corpus manifest: {}",
            missing.join(", ")
        )
    }
}

fn run_report(
    tasks: &[types::BenchmarkTask],
    corpus_path: &Path,
    tool_name: &str,
) -> Result<types::EvalReport> {
    Ok(match tool_name {
        "mock-perfect" => {
            let mock = runner::MockAdapter::perfect();
            runner::run_benchmark_with_mock(&mock, tasks)
        }
        "mock-partial" => {
            let mock = runner::MockAdapter::partial(0.7);
            runner::run_benchmark_with_mock(&mock, tasks)
        }
        "vera-bm25" => {
            let (repo_paths, corpus_shas) = load_verified_corpus(corpus_path)?;
            ensure_task_repos_known(tasks, &repo_paths)?;
            let vera = vera_adapter::VeraBm25Adapter::new()?;
            runner::run_benchmark(&vera, tasks, &repo_paths, &corpus_shas)
        }
        other => {
            anyhow::bail!(
                "Unknown tool '{}'. Available: vera-bm25, mock-perfect, mock-partial.",
                other
            );
        }
    })
}

fn print_retrieval_summary(label: &str, metrics: &types::RetrievalMetrics) {
    println!(
        "  {label}: R@1 {:.4}, R@5 {:.4}, R@10 {:.4}, MRR {:.4}, nDCG {:.4}",
        metrics.recall_at_1, metrics.recall_at_5, metrics.recall_at_10, metrics.mrr, metrics.ndcg
    );
}

fn cmd_stability(
    tasks_dir: &Path,
    corpus_path: &Path,
    tool_name: &str,
    tolerance: f64,
) -> Result<()> {
    let tasks = loader::load_tasks(tasks_dir)?;
    if tasks.is_empty() {
        anyhow::bail!("No benchmark tasks found");
    }

    eprintln!(
        "Running stability check with {} tasks via {}, tolerance ±{:.1}%",
        tasks.len(),
        tool_name,
        tolerance * 100.0
    );

    // Run 1
    let report1 = run_report(&tasks, corpus_path, tool_name)?;
    eprintln!(
        "  Run 1 complete: {} tasks evaluated",
        report1.per_task.len()
    );

    // Run 2
    let report2 = run_report(&tasks, corpus_path, tool_name)?;
    eprintln!(
        "  Run 2 complete: {} tasks evaluated",
        report2.per_task.len()
    );

    // Compare retrieval metrics
    let mut max_diff = 0.0f64;
    let mut violations = Vec::new();

    for (e1, e2) in report1.per_task.iter().zip(report2.per_task.iter()) {
        let metrics = [
            (
                "recall_at_1",
                e1.retrieval_metrics.recall_at_1,
                e2.retrieval_metrics.recall_at_1,
            ),
            (
                "recall_at_5",
                e1.retrieval_metrics.recall_at_5,
                e2.retrieval_metrics.recall_at_5,
            ),
            (
                "recall_at_10",
                e1.retrieval_metrics.recall_at_10,
                e2.retrieval_metrics.recall_at_10,
            ),
            ("mrr", e1.retrieval_metrics.mrr, e2.retrieval_metrics.mrr),
            ("ndcg", e1.retrieval_metrics.ndcg, e2.retrieval_metrics.ndcg),
        ];

        for (name, v1, v2) in metrics {
            let diff = (v1 - v2).abs();
            max_diff = max_diff.max(diff);
            if diff > tolerance {
                violations.push(format!(
                    "Task '{}' {}: run1={:.6}, run2={:.6}, diff={:.6} > tolerance {:.6}",
                    e1.task_id, name, v1, v2, diff, tolerance
                ));
            }
        }
    }

    // Compare latency (±10% tolerance)
    let lat_tolerance = 0.10;
    let lat1 = report1.aggregate.performance.latency_p50_ms;
    let lat2 = report2.aggregate.performance.latency_p50_ms;
    if lat1 > 0.0 {
        let lat_diff = ((lat1 - lat2) / lat1).abs();
        if lat_diff > lat_tolerance {
            eprintln!(
                "  Note: Latency p50 varied by {:.1}% (run1={:.2}ms, run2={:.2}ms) - within expected variance",
                lat_diff * 100.0,
                lat1,
                lat2
            );
        }
    }

    if violations.is_empty() {
        println!("✓ Stability check passed");
        println!(
            "  Max retrieval metric difference: {:.6} (tolerance: {:.6})",
            max_diff, tolerance
        );
        println!(
            "  All {} metrics across {} tasks within ±{:.1}%",
            5 * report1.per_task.len(),
            report1.per_task.len(),
            tolerance * 100.0
        );
        print_retrieval_summary("Run 1 aggregate", &report1.aggregate.retrieval);
        print_retrieval_summary("Run 2 aggregate", &report2.aggregate.retrieval);

        Ok(())
    } else {
        eprintln!("✗ Stability check FAILED:");
        for v in &violations {
            eprintln!("  - {v}");
        }
        std::process::exit(1);
    }
}
