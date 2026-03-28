//! `vera agent ...` — install and manage the Vera skill for coding agents.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use clap::ValueEnum;
use serde::Serialize;

use crate::skill_assets::{VERA_SKILL_FILES, VERA_SKILL_NAME};
use crate::state;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum AgentCommand {
    Install,
    Status,
    Remove,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AgentClient {
    /// Install for all supported clients at once.
    All,
    /// Cross-agent `.agents/skills/` directory (Agent Skills open spec).
    Agents,
    /// Sourcegraph Amp CLI.
    Amp,
    /// Antigravity CLI.
    Antigravity,
    /// Augment Code CLI.
    Augment,
    /// Anthropic Claude Code.
    Claude,
    /// Cline CLI.
    Cline,
    /// Codebuff CLI.
    Codebuff,
    /// CodeBuddy CLI.
    Codebuddy,
    /// OpenAI Codex CLI.
    Codex,
    /// GitHub Copilot CLI.
    Copilot,
    /// Snowflake Cortex Code.
    Cortex,
    /// Crush CLI.
    Crush,
    /// Cursor editor.
    Cursor,
    /// Factory Droid CLI.
    Droid,
    /// Google Gemini CLI.
    Gemini,
    /// Block Goose CLI.
    Goose,
    /// iFlow CLI.
    Iflow,
    /// JetBrains Junie CLI.
    Junie,
    /// Kilo Code CLI.
    Kilo,
    /// Kiro CLI.
    Kiro,
    /// Moonshot Kimi CLI.
    Kimi,
    /// Mistral Vibe CLI.
    Vibe,
    /// Mux CLI.
    Mux,
    /// OpenCode CLI.
    Opencode,
    /// OpenHands CLI.
    Openhands,
    /// Pi CLI.
    Pi,
    /// Qwen Code CLI.
    Qwen,
    /// Roo Code CLI.
    Roo,
    /// Trae CLI.
    Trae,
    /// Windsurf editor.
    Windsurf,
    /// Zed editor.
    Zed,
}

impl AgentClient {
    /// All concrete (non-All) client variants, in display order.
    fn all_concrete() -> &'static [AgentClient] {
        &[
            AgentClient::Agents,
            AgentClient::Amp,
            AgentClient::Antigravity,
            AgentClient::Augment,
            AgentClient::Claude,
            AgentClient::Cline,
            AgentClient::Codebuff,
            AgentClient::Codebuddy,
            AgentClient::Codex,
            AgentClient::Copilot,
            AgentClient::Cortex,
            AgentClient::Crush,
            AgentClient::Cursor,
            AgentClient::Droid,
            AgentClient::Gemini,
            AgentClient::Goose,
            AgentClient::Iflow,
            AgentClient::Junie,
            AgentClient::Kilo,
            AgentClient::Kiro,
            AgentClient::Kimi,
            AgentClient::Vibe,
            AgentClient::Mux,
            AgentClient::Opencode,
            AgentClient::Openhands,
            AgentClient::Pi,
            AgentClient::Qwen,
            AgentClient::Roo,
            AgentClient::Trae,
            AgentClient::Windsurf,
            AgentClient::Zed,
        ]
    }

    fn display_name(&self) -> &'static str {
        match self {
            AgentClient::All => "All",
            AgentClient::Agents => "Universal (.agents/skills/)",
            AgentClient::Amp => "Amp (Sourcegraph)",
            AgentClient::Antigravity => "Antigravity",
            AgentClient::Augment => "Augment Code",
            AgentClient::Claude => "Claude Code (Anthropic)",
            AgentClient::Cline => "Cline",
            AgentClient::Codebuff => "Codebuff",
            AgentClient::Codebuddy => "CodeBuddy",
            AgentClient::Codex => "Codex (OpenAI)",
            AgentClient::Copilot => "Copilot (GitHub)",
            AgentClient::Cortex => "Cortex Code (Snowflake)",
            AgentClient::Crush => "Crush",
            AgentClient::Cursor => "Cursor",
            AgentClient::Droid => "Droid (Factory)",
            AgentClient::Gemini => "Gemini CLI (Google)",
            AgentClient::Goose => "Goose (Block)",
            AgentClient::Iflow => "iFlow",
            AgentClient::Junie => "Junie (JetBrains)",
            AgentClient::Kilo => "Kilo Code",
            AgentClient::Kiro => "Kiro",
            AgentClient::Kimi => "Kimi (Moonshot)",
            AgentClient::Vibe => "Vibe (Mistral)",
            AgentClient::Mux => "Mux",
            AgentClient::Opencode => "OpenCode",
            AgentClient::Openhands => "OpenHands",
            AgentClient::Pi => "Pi",
            AgentClient::Qwen => "Qwen Code",
            AgentClient::Roo => "Roo Code",
            AgentClient::Trae => "Trae",
            AgentClient::Windsurf => "Windsurf",
            AgentClient::Zed => "Zed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AgentScope {
    Global,
    Project,
    All,
}

#[derive(Debug, Clone, Serialize)]
pub struct SkillLocationReport {
    client: AgentClient,
    scope: AgentScope,
    path: String,
    installed: bool,
}

pub fn run(
    command: AgentCommand,
    client: Option<AgentClient>,
    scope: Option<AgentScope>,
    json_output: bool,
) -> anyhow::Result<()> {
    match command {
        AgentCommand::Install => install(client, scope, json_output),
        AgentCommand::Status => status(
            client.unwrap_or(AgentClient::All),
            scope.unwrap_or(AgentScope::All),
            json_output,
        ),
        AgentCommand::Remove => remove(client, scope, json_output),
    }
}

fn install(
    client: Option<AgentClient>,
    scope: Option<AgentScope>,
    json_output: bool,
) -> anyhow::Result<()> {
    let (resolved_client, resolved_scope) = match (client, scope) {
        (Some(c), Some(s)) => (c, s),
        (Some(c), None) => (c, AgentScope::Global),
        (None, Some(s)) => (AgentClient::All, s),
        (None, None) if json_output => (AgentClient::All, AgentScope::Global),
        (None, None) => return install_interactive(),
    };

    let locations = resolve_locations(resolved_client, resolved_scope)?;
    do_install(&locations, json_output)
}

fn install_interactive() -> anyhow::Result<()> {
    cliclack::intro("vera agent install")?;

    let scope: AgentScope = cliclack::select("Install scope")
        .item(AgentScope::Global, "Global", "available in all projects")
        .item(AgentScope::Project, "Project", "current repo only")
        .item(AgentScope::All, "Both", "global and project")
        .interact()?;

    let all_clients = AgentClient::all_concrete();
    let mut multi = cliclack::multiselect("Select agents (space to toggle, all selected by default)");
    for &client in all_clients {
        multi = multi.item(client, client.display_name(), "");
    }
    let selected: Vec<AgentClient> = multi
        .initial_values(all_clients.to_vec())
        .required(true)
        .interact()?;

    let mut locations = Vec::new();
    let cwd = std::env::current_dir().context("failed to resolve current directory")?;
    let home = state::user_home_dir()?;
    for client in &selected {
        let scopes = match scope {
            AgentScope::All => vec![AgentScope::Global, AgentScope::Project],
            single => vec![single],
        };
        for s in scopes {
            locations.push(SkillLocation {
                client: *client,
                scope: s,
                path: skill_path_for(*client, s, &cwd, &home)?,
            });
        }
    }

    do_install(&locations, false)?;
    cliclack::outro("Done!")?;
    Ok(())
}

fn do_install(locations: &[SkillLocation], json_output: bool) -> anyhow::Result<()> {
    for location in locations {
        if location.path.exists() {
            fs::remove_dir_all(&location.path).with_context(|| {
                format!(
                    "failed to replace existing skill at {}",
                    location.path.display()
                )
            })?;
        }
        install_skill_to(&location.path)?;
    }

    let reports: Vec<SkillLocationReport> = locations
        .iter()
        .map(|location| SkillLocationReport {
            client: location.client,
            scope: location.scope,
            path: location.path.display().to_string(),
            installed: true,
        })
        .collect();

    if json_output {
        println!("{}", serde_json::to_string_pretty(&reports)?);
    } else {
        println!("Installed Vera skill:");
        println!();
        for report in &reports {
            println!(
                "  {:<14} {:<7} {}",
                format!("{:?}", report.client).to_lowercase(),
                format!("{:?}", report.scope).to_lowercase(),
                report.path
            );
        }
    }

    Ok(())
}

fn remove(
    client: Option<AgentClient>,
    scope: Option<AgentScope>,
    json_output: bool,
) -> anyhow::Result<()> {
    let (resolved_client, resolved_scope) = match (client, scope) {
        (Some(c), Some(s)) => (c, s),
        (Some(c), None) => (c, AgentScope::Global),
        (None, Some(s)) => (AgentClient::All, s),
        (None, None) if json_output => (AgentClient::All, AgentScope::All),
        (None, None) => return remove_interactive(),
    };

    let locations = resolve_locations(resolved_client, resolved_scope)?;
    do_remove(&locations, json_output)
}

fn remove_interactive() -> anyhow::Result<()> {
    let cwd = std::env::current_dir().context("failed to resolve current directory")?;
    let home = state::user_home_dir()?;
    let all_clients = AgentClient::all_concrete();

    let mut installed: Vec<(AgentClient, AgentScope, PathBuf)> = Vec::new();
    for &client in all_clients {
        for scope in [AgentScope::Global, AgentScope::Project] {
            let path = skill_path_for(client, scope, &cwd, &home)?;
            if path.join("SKILL.md").exists() {
                installed.push((client, scope, path));
            }
        }
    }

    if installed.is_empty() {
        println!("No Vera skill installations found.");
        return Ok(());
    }

    cliclack::intro("vera agent remove")?;

    let mut multi = cliclack::multiselect("Select installations to remove");
    for (i, (c, s, p)) in installed.iter().enumerate() {
        let label = format!(
            "{} ({})",
            c.display_name(),
            format!("{:?}", s).to_lowercase()
        );
        let hint = p.display().to_string();
        multi = multi.item(i, label, hint);
    }
    let selected: Vec<usize> = multi.required(true).interact()?;

    let locations: Vec<SkillLocation> = selected
        .iter()
        .map(|&idx| {
            let (client, scope, path) = &installed[idx];
            SkillLocation {
                client: *client,
                scope: *scope,
                path: path.clone(),
            }
        })
        .collect();

    do_remove(&locations, false)?;
    cliclack::outro("Done!")?;
    Ok(())
}

fn do_remove(locations: &[SkillLocation], json_output: bool) -> anyhow::Result<()> {
    let mut reports = Vec::with_capacity(locations.len());

    for location in locations {
        let installed = location.path.join("SKILL.md").exists();
        if installed {
            fs::remove_dir_all(&location.path).with_context(|| {
                format!(
                    "failed to remove installed skill at {}",
                    location.path.display()
                )
            })?;
        }
        reports.push(SkillLocationReport {
            client: location.client,
            scope: location.scope,
            path: location.path.display().to_string(),
            installed: false,
        });
    }

    if json_output {
        println!("{}", serde_json::to_string_pretty(&reports)?);
    } else {
        println!("Removed Vera skill from:");
        println!();
        for report in &reports {
            println!(
                "  {:<14} {:<7} {}",
                format!("{:?}", report.client).to_lowercase(),
                format!("{:?}", report.scope).to_lowercase(),
                report.path
            );
        }
    }

    Ok(())
}

fn status(client: AgentClient, scope: AgentScope, json_output: bool) -> anyhow::Result<()> {
    let reports = resolve_locations(client, scope)?
        .into_iter()
        .map(|location| SkillLocationReport {
            client: location.client,
            scope: location.scope,
            path: location.path.display().to_string(),
            installed: location.path.join("SKILL.md").exists(),
        })
        .collect::<Vec<_>>();

    if json_output {
        println!("{}", serde_json::to_string_pretty(&reports)?);
    } else {
        println!("Vera skill status:");
        println!();
        for report in &reports {
            let status = if report.installed {
                "installed"
            } else {
                "missing"
            };
            println!(
                "  {:<14} {:<7} {:<10} {}",
                format!("{:?}", report.client).to_lowercase(),
                format!("{:?}", report.scope).to_lowercase(),
                status,
                report.path
            );
        }
    }

    Ok(())
}

fn install_skill_to(target_dir: &Path) -> anyhow::Result<()> {
    for file in VERA_SKILL_FILES {
        let path = target_dir.join(file.relative_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        fs::write(&path, file.contents)
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    let version_path = target_dir.join(".version");
    fs::write(&version_path, env!("CARGO_PKG_VERSION"))
        .with_context(|| format!("failed to write {}", version_path.display()))?;
    Ok(())
}

#[derive(Debug, Clone)]
struct SkillLocation {
    client: AgentClient,
    scope: AgentScope,
    path: PathBuf,
}

fn resolve_locations(client: AgentClient, scope: AgentScope) -> anyhow::Result<Vec<SkillLocation>> {
    let cwd = std::env::current_dir().context("failed to resolve current directory")?;
    let home = state::user_home_dir()?;
    resolve_locations_with_roots(client, scope, &cwd, &home)
}

fn resolve_locations_with_roots(
    client: AgentClient,
    scope: AgentScope,
    cwd: &Path,
    home: &Path,
) -> anyhow::Result<Vec<SkillLocation>> {
    let clients = match client {
        AgentClient::All => AgentClient::all_concrete().to_vec(),
        single => vec![single],
    };
    let scopes = match scope {
        AgentScope::All => vec![AgentScope::Global, AgentScope::Project],
        single => vec![single],
    };

    let mut locations = Vec::new();
    for client in clients {
        for scope in &scopes {
            locations.push(SkillLocation {
                client,
                scope: *scope,
                path: skill_path_for(client, *scope, cwd, home)?,
            });
        }
    }

    Ok(locations)
}

fn skill_path_for(
    client: AgentClient,
    scope: AgentScope,
    cwd: &Path,
    home: &Path,
) -> anyhow::Result<PathBuf> {
    if scope == AgentScope::All {
        bail!("scope=all is only valid before path resolution");
    }

    let base = match (client, scope) {
        (AgentClient::Agents, AgentScope::Global) => {
            home.join(".config").join("agents").join("skills")
        }
        (AgentClient::Agents, AgentScope::Project) => cwd.join(".agents").join("skills"),
        (AgentClient::Amp, AgentScope::Global) => {
            home.join(".config").join("agents").join("skills")
        }
        (AgentClient::Amp, AgentScope::Project) => cwd.join(".agents").join("skills"),
        (AgentClient::Antigravity, AgentScope::Global) => {
            home.join(".gemini").join("antigravity").join("skills")
        }
        (AgentClient::Antigravity, AgentScope::Project) => cwd.join(".agent").join("skills"),
        (AgentClient::Augment, AgentScope::Global) => home.join(".augment").join("skills"),
        (AgentClient::Augment, AgentScope::Project) => cwd.join(".augment").join("skills"),
        (AgentClient::Claude, AgentScope::Global) => home.join(".claude").join("skills"),
        (AgentClient::Claude, AgentScope::Project) => cwd.join(".claude").join("skills"),
        (AgentClient::Cline, AgentScope::Global) => home.join(".agents").join("skills"),
        (AgentClient::Cline, AgentScope::Project) => cwd.join(".agents").join("skills"),
        (AgentClient::Codebuff, AgentScope::Global) => home.join(".codebuff").join("skills"),
        (AgentClient::Codebuff, AgentScope::Project) => cwd.join(".codebuff").join("skills"),
        (AgentClient::Codebuddy, AgentScope::Global) => home.join(".codebuddy").join("skills"),
        (AgentClient::Codebuddy, AgentScope::Project) => cwd.join(".codebuddy").join("skills"),
        (AgentClient::Codex, AgentScope::Global) => home.join(".codex").join("skills"),
        (AgentClient::Codex, AgentScope::Project) => cwd.join(".agents").join("skills"),
        (AgentClient::Copilot, AgentScope::Global) => home.join(".copilot").join("skills"),
        (AgentClient::Copilot, AgentScope::Project) => cwd.join(".agents").join("skills"),
        (AgentClient::Cortex, AgentScope::Global) => {
            home.join(".snowflake").join("cortex").join("skills")
        }
        (AgentClient::Cortex, AgentScope::Project) => cwd.join(".cortex").join("skills"),
        (AgentClient::Crush, AgentScope::Global) => {
            home.join(".config").join("crush").join("skills")
        }
        (AgentClient::Crush, AgentScope::Project) => cwd.join(".crush").join("skills"),
        (AgentClient::Cursor, AgentScope::Global) => home.join(".cursor").join("skills"),
        (AgentClient::Cursor, AgentScope::Project) => cwd.join(".agents").join("skills"),
        (AgentClient::Droid, AgentScope::Global) => home.join(".factory").join("skills"),
        (AgentClient::Droid, AgentScope::Project) => cwd.join(".factory").join("skills"),
        (AgentClient::Gemini, AgentScope::Global) => home.join(".gemini").join("skills"),
        (AgentClient::Gemini, AgentScope::Project) => cwd.join(".agents").join("skills"),
        (AgentClient::Goose, AgentScope::Global) => {
            home.join(".config").join("goose").join("skills")
        }
        (AgentClient::Goose, AgentScope::Project) => cwd.join(".goose").join("skills"),
        (AgentClient::Iflow, AgentScope::Global) => home.join(".iflow").join("skills"),
        (AgentClient::Iflow, AgentScope::Project) => cwd.join(".iflow").join("skills"),
        (AgentClient::Junie, AgentScope::Global) => home.join(".junie").join("skills"),
        (AgentClient::Junie, AgentScope::Project) => cwd.join(".junie").join("skills"),
        (AgentClient::Kilo, AgentScope::Global) => home.join(".kilocode").join("skills"),
        (AgentClient::Kilo, AgentScope::Project) => cwd.join(".kilocode").join("skills"),
        (AgentClient::Kiro, AgentScope::Global) => home.join(".kiro").join("skills"),
        (AgentClient::Kiro, AgentScope::Project) => cwd.join(".kiro").join("skills"),
        (AgentClient::Kimi, AgentScope::Global) => {
            home.join(".config").join("agents").join("skills")
        }
        (AgentClient::Kimi, AgentScope::Project) => cwd.join(".agents").join("skills"),
        (AgentClient::Vibe, AgentScope::Global) => home.join(".vibe").join("skills"),
        (AgentClient::Vibe, AgentScope::Project) => cwd.join(".vibe").join("skills"),
        (AgentClient::Mux, AgentScope::Global) => home.join(".mux").join("skills"),
        (AgentClient::Mux, AgentScope::Project) => cwd.join(".mux").join("skills"),
        (AgentClient::Opencode, AgentScope::Global) => {
            home.join(".config").join("opencode").join("skills")
        }
        (AgentClient::Opencode, AgentScope::Project) => cwd.join(".agents").join("skills"),
        (AgentClient::Openhands, AgentScope::Global) => home.join(".openhands").join("skills"),
        (AgentClient::Openhands, AgentScope::Project) => cwd.join(".openhands").join("skills"),
        (AgentClient::Pi, AgentScope::Global) => home.join(".pi").join("agent").join("skills"),
        (AgentClient::Pi, AgentScope::Project) => cwd.join(".pi").join("skills"),
        (AgentClient::Qwen, AgentScope::Global) => home.join(".qwen").join("skills"),
        (AgentClient::Qwen, AgentScope::Project) => cwd.join(".qwen").join("skills"),
        (AgentClient::Roo, AgentScope::Global) => home.join(".roo").join("skills"),
        (AgentClient::Roo, AgentScope::Project) => cwd.join(".roo").join("skills"),
        (AgentClient::Trae, AgentScope::Global) => home.join(".trae").join("skills"),
        (AgentClient::Trae, AgentScope::Project) => cwd.join(".trae").join("skills"),
        (AgentClient::Windsurf, AgentScope::Global) => {
            home.join(".codeium").join("windsurf").join("skills")
        }
        (AgentClient::Windsurf, AgentScope::Project) => cwd.join(".windsurf").join("skills"),
        (AgentClient::Zed, AgentScope::Global) => home.join(".zed").join("skills"),
        (AgentClient::Zed, AgentScope::Project) => cwd.join(".zed").join("skills"),
        (AgentClient::All, _) => bail!("client=all is only valid before path resolution"),
        (_, AgentScope::All) => bail!("scope=all is only valid before path resolution"),
    };

    Ok(base.join(VERA_SKILL_NAME))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_locations_expands_scope_all() {
        let cwd = Path::new("/tmp/project");
        let home = Path::new("/tmp/home");
        let locations =
            resolve_locations_with_roots(AgentClient::Codex, AgentScope::All, cwd, home).unwrap();

        assert_eq!(locations.len(), 2);
        assert!(
            locations
                .iter()
                .any(|location| location.scope == AgentScope::Global)
        );
        assert!(
            locations
                .iter()
                .any(|location| location.scope == AgentScope::Project)
        );
    }

    #[test]
    fn copilot_project_skill_uses_agents_dir() {
        let cwd = Path::new("/tmp/project");
        let home = Path::new("/tmp/home");
        let path = skill_path_for(AgentClient::Copilot, AgentScope::Project, cwd, home).unwrap();
        assert_eq!(path, PathBuf::from("/tmp/project/.agents/skills/vera"));
    }

    #[test]
    fn all_concrete_clients_have_paths() {
        let cwd = Path::new("/tmp/project");
        let home = Path::new("/tmp/home");
        for &client in AgentClient::all_concrete() {
            skill_path_for(client, AgentScope::Global, cwd, home)
                .unwrap_or_else(|_| panic!("no global path for {:?}", client));
            skill_path_for(client, AgentScope::Project, cwd, home)
                .unwrap_or_else(|_| panic!("no project path for {:?}", client));
        }
    }
}
