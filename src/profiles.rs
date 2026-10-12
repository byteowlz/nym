//! Redaction profiles: named, private term lists shared by every command.
//!
//! `$XDG_CONFIG_HOME/nym/profiles/<name>/{sensitive,allow}.txt`, one literal
//! per line (`#` comments). The `default` profile applies whenever it exists;
//! named profiles add to it. Allow lists never veto credentials, local
//! identity or sensitive terms.
use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand, ValueEnum};
use serde::Serialize;
use std::{
    collections::BTreeSet,
    fs,
    io::Write,
    path::{Path, PathBuf},
};

pub const DEFAULT: &str = "default";

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum List {
    /// Always replaced wherever they occur
    Sensitive,
    /// Never replaced as detector findings
    Allow,
}

impl List {
    fn file(self) -> &'static str {
        match self {
            List::Sensitive => "sensitive.txt",
            List::Allow => "allow.txt",
        }
    }
}

/// Terms from the default profile plus the requested profiles.
#[derive(Default, Debug)]
pub struct Lists {
    pub sensitive: Vec<String>,
    pub allow: Vec<String>,
}

pub fn root() -> Result<PathBuf> {
    crate::config::global_config_path()
        .and_then(|p| p.parent().map(|d| d.join("profiles")))
        .context("no configuration directory for redaction profiles")
}

fn valid_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > 64
        || !name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
    {
        bail!("profile names use lowercase letters, digits, '-' and '_' (max 64)");
    }
    Ok(())
}

fn dir(root: &Path, name: &str) -> Result<PathBuf> {
    valid_name(name)?;
    Ok(root.join(name))
}

fn read(path: &Path) -> Result<Vec<String>> {
    match fs::read_to_string(path) {
        Ok(text) => Ok(parse(&text)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(_) => bail!("failed to read a redaction profile list"),
    }
}

fn parse(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect()
}

/// Load `default` (when present) plus every named profile; named ones must exist.
pub fn load(names: &[String]) -> Result<Lists> {
    load_from(&root()?, names)
}

fn load_from(root: &Path, names: &[String]) -> Result<Lists> {
    let mut selected = vec![DEFAULT.to_string()];
    for name in names {
        valid_name(name)?;
        if !root.join(name).is_dir() {
            bail!("redaction profile '{name}' does not exist; create it with `nym profiles add`");
        }
        if !selected.contains(name) {
            selected.push(name.clone());
        }
    }
    let mut lists = Lists::default();
    for name in &selected {
        let base = dir(root, name)?;
        for term in read(&base.join(List::Sensitive.file()))? {
            if !lists.sensitive.contains(&term) {
                lists.sensitive.push(term);
            }
        }
        for term in read(&base.join(List::Allow.file()))? {
            if !lists.allow.contains(&term) {
                lists.allow.push(term);
            }
        }
    }
    Ok(lists)
}

fn private_dir(path: &Path) -> Result<()> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder
        .create(path)
        .context("failed to create redaction profile directory")
}

/// Atomic owner-only rewrite, sorted and deduplicated.
fn write(path: &Path, terms: &BTreeSet<String>) -> Result<()> {
    let parent = path.parent().context("invalid profile path")?;
    private_dir(parent)?;
    let mut tmp = tempfile::Builder::new()
        .prefix(".nym-profile-")
        .tempfile_in(parent)?;
    for term in terms {
        writeln!(tmp, "{term}")?;
    }
    tmp.as_file().sync_all()?;
    tmp.persist(path).map_err(|e| e.error)?;
    Ok(())
}

#[derive(Args, Debug)]
pub struct ProfilesCommand {
    #[command(subcommand)]
    action: Action,
}

#[derive(Subcommand, Debug)]
enum Action {
    /// List profiles with term counts (no terms are printed)
    List,
    /// Show one profile's counts and location; --terms prints the terms
    Show {
        name: String,
        #[arg(long)]
        terms: bool,
    },
    /// Add terms to a profile list (creates the profile)
    Add {
        name: String,
        #[arg(long, value_enum)]
        list: List,
        #[arg(required = true)]
        terms: Vec<String>,
    },
    /// Remove terms from a profile list
    Remove {
        name: String,
        #[arg(long, value_enum)]
        list: List,
        #[arg(required = true)]
        terms: Vec<String>,
    },
    /// Merge a term file (e.g. the review page's trace-allow.txt) into a profile list
    Import {
        name: String,
        #[arg(long, value_enum)]
        list: List,
        file: PathBuf,
    },
    /// Print a profile's directory, for editing the lists by hand
    Path { name: Option<String> },
}

#[derive(Serialize)]
struct Summary {
    name: String,
    path: PathBuf,
    sensitive: usize,
    allow: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    sensitive_terms: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    allow_terms: Option<Vec<String>>,
}

fn summary(root: &Path, name: &str, terms: bool) -> Result<Summary> {
    let base = dir(root, name)?;
    let sensitive = read(&base.join(List::Sensitive.file()))?;
    let allow = read(&base.join(List::Allow.file()))?;
    Ok(Summary {
        name: name.to_string(),
        path: base,
        sensitive: sensitive.len(),
        allow: allow.len(),
        sensitive_terms: terms.then_some(sensitive),
        allow_terms: terms.then_some(allow),
    })
}

fn update(
    root: &Path,
    name: &str,
    list: List,
    change: impl FnOnce(&mut BTreeSet<String>),
) -> Result<usize> {
    let path = dir(root, name)?.join(list.file());
    let mut terms: BTreeSet<String> = read(&path)?.into_iter().collect();
    change(&mut terms);
    write(&path, &terms)?;
    Ok(terms.len())
}

pub fn run(common: &crate::CommonOpts, cmd: ProfilesCommand) -> Result<()> {
    run_in(common, &root()?, cmd)
}

fn run_in(common: &crate::CommonOpts, root: &Path, cmd: ProfilesCommand) -> Result<()> {
    let emit = |value: &dyn erased::Emit, text: String| -> Result<()> {
        if common.json {
            println!("{}", value.json()?);
        } else if !common.quiet {
            println!("{text}");
        }
        Ok(())
    };
    match cmd.action {
        Action::List => {
            let mut names = Vec::new();
            if let Ok(entries) = fs::read_dir(root) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    if entry.path().is_dir() && valid_name(&name).is_ok() {
                        names.push(name);
                    }
                }
            }
            names.sort();
            let rows = names
                .iter()
                .map(|n| summary(root, n, false))
                .collect::<Result<Vec<_>>>()?;
            let text = if rows.is_empty() {
                format!(
                    "No redaction profiles yet ({}). Create one with `nym profiles add default --list allow TERM`.",
                    root.display()
                )
            } else {
                rows.iter()
                    .map(|r| {
                        format!(
                            "{:<16} {:>5} sensitive {:>5} allow{}",
                            r.name,
                            r.sensitive,
                            r.allow,
                            if r.name == DEFAULT {
                                "  (always applied)"
                            } else {
                                ""
                            }
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            emit(&rows, text)
        }
        Action::Show { name, terms } => {
            let s = summary(root, &name, terms)?;
            let mut text = format!(
                "{}: {} sensitive, {} allow\n{}",
                s.name,
                s.sensitive,
                s.allow,
                s.path.display()
            );
            if let (Some(sens), Some(allow)) = (&s.sensitive_terms, &s.allow_terms) {
                text.push_str(&format!(
                    "\n[sensitive]\n{}\n[allow]\n{}",
                    sens.join("\n"),
                    allow.join("\n")
                ));
            }
            emit(&s, text)
        }
        Action::Add { name, list, terms } => {
            let terms: Vec<String> = terms
                .iter()
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
                .collect();
            let total = update(root, &name, list, |set| set.extend(terms.iter().cloned()))?;
            emit(
                &summary(root, &name, false)?,
                format!("{name}: {} {list:?} terms ({} given)", total, terms.len()).to_lowercase(),
            )
        }
        Action::Remove { name, list, terms } => {
            if !dir(root, &name)?.is_dir() {
                bail!("redaction profile '{name}' does not exist");
            }
            let total = update(root, &name, list, |set| {
                for t in &terms {
                    set.remove(t.trim());
                }
            })?;
            emit(
                &summary(root, &name, false)?,
                format!("{name}: {total} {list:?} terms").to_lowercase(),
            )
        }
        Action::Import { name, list, file } => {
            let text = fs::read_to_string(&file).context("failed to read the term file")?;
            let incoming = parse(&text);
            let total = update(root, &name, list, |set| {
                set.extend(incoming.iter().cloned())
            })?;
            emit(
                &summary(root, &name, false)?,
                format!(
                    "{name}: {total} {list:?} terms after importing {}",
                    incoming.len()
                )
                .to_lowercase(),
            )
        }
        Action::Path { name } => {
            let path = match name {
                Some(n) => dir(root, &n)?,
                None => root.to_path_buf(),
            };
            println!("{}", path.display());
            Ok(())
        }
    }
}

/// Minimal object-safe JSON emission for heterogeneous summaries.
mod erased {
    pub trait Emit {
        fn json(&self) -> serde_json::Result<String>;
    }
    impl<T: serde::Serialize> Emit for T {
        fn json(&self) -> serde_json::Result<String> {
            serde_json::to_string_pretty(self)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_always_applies_and_named_profiles_add() {
        let dir = tempfile::TempDir::new().unwrap();
        let root = dir.path();
        update(root, DEFAULT, List::Allow, |s| {
            s.insert("Docker".into());
        })
        .unwrap();
        update(root, "work", List::Sensitive, |s| {
            s.extend(["Acme Corp".to_string(), "acme-internal".to_string()]);
        })
        .unwrap();
        fs::write(
            root.join("work/allow.txt"),
            "# comment\n  oqto  \n\nDocker\n",
        )
        .unwrap();
        let base = load_from(root, &[]).unwrap();
        assert_eq!(
            (base.sensitive, base.allow),
            (Vec::<String>::new(), vec!["Docker".to_string()])
        );
        let work = load_from(root, &["work".into()]).unwrap();
        assert_eq!(
            work.sensitive,
            vec!["Acme Corp".to_string(), "acme-internal".to_string()]
        );
        assert_eq!(work.allow, vec!["Docker".to_string(), "oqto".to_string()]);
        assert!(load_from(root, &["missing".into()]).is_err());
        assert!(load_from(root, &["../escape".into()]).is_err());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(root.join("work"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o700
            );
            assert_eq!(
                fs::metadata(root.join("work/sensitive.txt"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o077,
                0
            );
        }
    }
}
