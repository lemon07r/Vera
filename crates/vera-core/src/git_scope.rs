//! Git-aware path scoping for changed-file workflows.

use std::collections::HashSet;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};

/// Restrict a command to files selected from git state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GitScope {
    /// Modified, staged, and untracked files in the current working tree.
    Changed,
    /// Files changed since a specific revision.
    Since(String),
    /// Files changed since `merge-base(HEAD, rev)`.
    Base(String),
}

/// Resolve a git scope to an exact set of repository-relative paths.
pub fn resolve_scope(repo_root: &Path, scope: &GitScope) -> Result<HashSet<String>> {
    ensure_git_repo(repo_root)?;

    match scope {
        GitScope::Changed => resolve_changed_files(repo_root),
        GitScope::Since(rev) => resolve_diff_files(repo_root, rev),
        GitScope::Base(rev) => {
            let merge_base = run_git(repo_root, &["merge-base", "HEAD", rev])?;
            let merge_base = merge_base.trim();
            if merge_base.is_empty() {
                bail!("git merge-base returned an empty revision for {rev}");
            }
            resolve_diff_files(repo_root, merge_base)
        }
    }
}

fn ensure_git_repo(repo_root: &Path) -> Result<()> {
    let _ = run_git(repo_root, &["rev-parse", "--show-toplevel"])
        .context("failed to verify git repository for changed-file scope")?;
    Ok(())
}

fn resolve_changed_files(repo_root: &Path) -> Result<HashSet<String>> {
    let mut files = HashSet::new();
    for args in [
        vec![
            "diff",
            "--name-only",
            "--diff-filter=ACMRTUXB",
            "--cached",
            "--",
        ],
        vec!["diff", "--name-only", "--diff-filter=ACMRTUXB", "--"],
        vec!["ls-files", "--others", "--exclude-standard", "--"],
    ] {
        files.extend(parse_git_paths(&run_git(repo_root, &args)?));
    }
    Ok(files)
}

fn resolve_diff_files(repo_root: &Path, revision: &str) -> Result<HashSet<String>> {
    let output = run_git(
        repo_root,
        &[
            "diff",
            "--name-only",
            "--diff-filter=ACMRTUXB",
            revision,
            "--",
        ],
    )?;
    Ok(parse_git_paths(&output))
}

fn run_git(repo_root: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo_root)
        .output()
        .with_context(|| format!("failed to run git {}", args.join(" ")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let details = if stderr.is_empty() {
            format!(
                "git {} exited with status {}",
                args.join(" "),
                output.status
            )
        } else {
            stderr
        };
        bail!(details);
    }

    String::from_utf8(output.stdout)
        .with_context(|| format!("git {} produced non-UTF-8 output", args.join(" ")))
}

fn parse_git_paths(output: &str) -> HashSet<String> {
    output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| line.replace('\\', "/"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;
    use tempfile::TempDir;

    #[test]
    fn parse_git_paths_skips_empty_lines() {
        let paths = parse_git_paths("src/main.rs\n\nREADME.md\n");
        assert!(paths.contains("src/main.rs"));
        assert!(paths.contains("README.md"));
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn resolve_changed_scope_includes_modified_and_untracked_files() {
        let repo = init_repo();
        std::fs::write(
            repo.path().join("tracked.rs"),
            "fn tracked() { println!(\"changed\"); }\n",
        )
        .unwrap();
        std::fs::write(repo.path().join("new.py"), "def created():\n    pass\n").unwrap();

        let paths = resolve_scope(repo.path(), &GitScope::Changed).unwrap();
        assert!(paths.contains("tracked.rs"));
        assert!(paths.contains("new.py"));
    }

    #[test]
    fn resolve_since_scope_uses_revision_diff() {
        let repo = init_repo();
        let head = run_git(repo.path(), &["rev-parse", "HEAD"]).unwrap();
        std::fs::write(
            repo.path().join("tracked.rs"),
            "fn tracked() { println!(\"changed\"); }\n",
        )
        .unwrap();

        let paths = resolve_scope(repo.path(), &GitScope::Since(head.trim().to_string())).unwrap();
        assert!(paths.contains("tracked.rs"));
    }

    #[test]
    fn resolve_base_scope_uses_merge_base() {
        let repo = init_repo();
        run_git(repo.path(), &["checkout", "-b", "feature"]).unwrap();
        std::fs::write(
            repo.path().join("tracked.rs"),
            "fn tracked() { println!(\"feature\"); }\n",
        )
        .unwrap();
        commit_all(repo.path(), "feature commit");

        let paths = resolve_scope(repo.path(), &GitScope::Base("HEAD~1".to_string())).unwrap();
        assert!(paths.contains("tracked.rs"));
    }

    fn init_repo() -> TempDir {
        let repo = TempDir::new().unwrap();
        git(repo.path(), &["init"]).unwrap();
        std::fs::write(repo.path().join("tracked.rs"), "fn tracked() {}\n").unwrap();
        commit_all(repo.path(), "initial commit");
        repo
    }

    fn commit_all(repo_root: &Path, message: &str) {
        git(repo_root, &["add", "."]).unwrap();
        git(
            repo_root,
            &[
                "-c",
                "user.name=Vera Test",
                "-c",
                "user.email=vera@example.com",
                "commit",
                "-m",
                message,
            ],
        )
        .unwrap();
    }

    fn git(repo_root: &Path, args: &[&str]) -> Result<()> {
        let status = Command::new("git")
            .args(args)
            .current_dir(repo_root)
            .status()
            .with_context(|| format!("failed to run git {}", args.join(" ")))?;
        if !status.success() {
            bail!("git {} failed with status {}", args.join(" "), status);
        }
        Ok(())
    }
}
