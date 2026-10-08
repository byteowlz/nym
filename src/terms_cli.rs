//! CLI and private artifact publication for model-independent vocabulary review.
use crate::{
    engine::selector::PathSelector,
    input::{self, FormatArg},
    terms::{self, Builder, Choice, Decision, Discovery, Review, Settings},
};
use anyhow::{Result, bail};
use clap::{Args, Subcommand};
use serde::Serialize;
use serde_json::Value;
use std::{
    fs,
    io::{BufRead, Read, Write},
    path::{Path, PathBuf},
};

#[derive(Args)]
pub struct TermsCommand {
    #[command(subcommand)]
    action: Action,
}

impl std::fmt::Debug for TermsCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TermsCommand").finish_non_exhaustive()
    }
}

#[derive(Subcommand)]
enum Action {
    /// Discover recurring literal candidates; never automatically classify privacy
    Discover {
        /// Input files; omit to read stdin. Directories and binary documents are not supported
        inputs: Vec<PathBuf>,
        #[arg(long, value_enum)]
        format: Option<FormatArg>,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long)]
        include: Vec<String>,
        #[arg(long)]
        exclude: Vec<String>,
        #[arg(long)]
        min_count: Option<u64>,
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long)]
        max_distinct: Option<usize>,
        #[arg(long)]
        phrase_words: Option<usize>,
        #[arg(long)]
        max_unit_bytes: Option<usize>,
        /// JSON map of public/reference literal frequency counts; ranking only
        #[arg(long)]
        background: Option<PathBuf>,
        #[arg(long)]
        force: bool,
    },
    /// Create or update source-bound decisions, or emit a self-contained browser review
    Review {
        discovery: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        /// Existing decisions to retain; must match the exact discovery artifact
        #[arg(long)]
        resume: Option<PathBuf>,
        /// Explicit ID=sensitive|contextual|dismiss|unsure; repeatable
        #[arg(long)]
        decision: Vec<String>,
        /// Write an offline, phone-sized HTML review instead of JSON; no server or uploads
        #[arg(long, conflicts_with = "decision")]
        html: bool,
        #[arg(long)]
        force: bool,
    },
    /// Export only explicitly approved literals for --sensitive-terms-file
    Export {
        discovery: PathBuf,
        #[arg(long)]
        review: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        force: bool,
    },
}

#[derive(Serialize)]
struct Summary {
    action: &'static str,
    candidates: usize,
    approved: usize,
    units: u64,
    pending: usize,
}

pub fn run(common: &crate::CommonOpts, defaults: &Settings, command: TermsCommand) -> Result<()> {
    let summary = match command.action {
        Action::Discover {
            inputs,
            format,
            output,
            include,
            exclude,
            min_count,
            limit,
            max_distinct,
            phrase_words,
            max_unit_bytes,
            background,
            force,
        } => {
            let selector = PathSelector::new(&include, &exclude)
                .map_err(|_| anyhow::anyhow!("invalid discovery selector"))?;
            let background = background.map(|p| read_json(&p)).transpose()?;
            let settings = Settings {
                max_distinct: max_distinct.unwrap_or(defaults.max_distinct),
                min_count: min_count.unwrap_or(defaults.min_count),
                max_candidates: limit.unwrap_or(defaults.max_candidates),
                max_phrase_words: phrase_words.unwrap_or(defaults.max_phrase_words),
                max_unit_bytes: max_unit_bytes.unwrap_or(defaults.max_unit_bytes),
                ..defaults.clone()
            };
            let max_unit_bytes = settings.max_unit_bytes;
            let mut builder = Builder::new(settings, background)?;
            if inputs.is_empty() {
                collect_input(&mut builder, None, format, &selector, max_unit_bytes)?;
            } else {
                for path in &inputs {
                    collect_input(&mut builder, Some(path), format, &selector, max_unit_bytes)?;
                }
            }
            let discovery = builder.finish()?;
            write_json(&output, &discovery, force)?;
            Summary {
                action: "discover",
                candidates: discovery.candidates.len(),
                approved: 0,
                units: discovery.unit_count,
                pending: discovery.candidates.len(),
            }
        }
        Action::Review {
            discovery,
            output,
            resume,
            decision,
            html,
            force,
        } => {
            let discovery: Discovery = read_json(&discovery)?;
            discovery.validate()?;
            let mut review = if let Some(path) = resume {
                read_json::<Review>(&path)?
            } else {
                Review::new(&discovery)?
            };
            review.validate(&discovery)?;
            for value in decision {
                let Some((id, choice)) = value.split_once('=') else {
                    bail!("decision must be ID=choice");
                };
                let choice = match choice {
                    "sensitive" => Choice::Sensitive,
                    "contextual" => Choice::Contextual,
                    "dismiss" => Choice::Dismiss,
                    "unsure" => Choice::Unsure,
                    _ => bail!("decision choice must be sensitive, contextual, dismiss or unsure"),
                };
                if let Some(row) = review.decisions.iter_mut().find(|d| d.candidate_id == id) {
                    row.choice = choice;
                } else {
                    review.decisions.push(Decision {
                        candidate_id: id.to_owned(),
                        choice,
                    });
                }
            }
            review.validate(&discovery)?;
            if html {
                let data = serde_json::to_string(
                    &serde_json::json!({"discovery":discovery,"review":review}),
                )
                .map_err(|_| anyhow::anyhow!("review serialization failed"))?
                .replace('&', "\\u0026")
                .replace('<', "\\u003c")
                .replace('>', "\\u003e");
                let document = include_str!("terms_review.html").replace("__NYM_DATA__", &data);
                private_write(&output, document.as_bytes(), force)?;
            } else {
                write_json(&output, &review, force)?;
            }
            Summary {
                action: "review",
                candidates: discovery.candidates.len(),
                approved: review.approve_terms(&discovery)?.len(),
                units: discovery.unit_count,
                pending: discovery.candidates.len() - review.decisions.len(),
            }
        }
        Action::Export {
            discovery,
            review,
            output,
            dry_run,
            force,
        } => {
            let discovery: Discovery = read_json(&discovery)?;
            let review: Review = read_json(&review)?;
            let approved = review.approve_terms(&discovery)?;
            if approved.is_empty() {
                bail!("no explicitly approved sensitive terms to export");
            }
            if !dry_run {
                private_write(
                    &output,
                    format!("{}\n", approved.join("\n")).as_bytes(),
                    force,
                )?;
            }
            Summary {
                action: "export",
                candidates: discovery.candidates.len(),
                approved: approved.len(),
                units: discovery.unit_count,
                pending: discovery.candidates.len() - review.decisions.len(),
            }
        }
    };
    if !common.quiet {
        if common.json {
            println!("{}", serde_json::to_string(&summary)?);
        } else if common.yaml {
            print!("{}", serde_yaml::to_string(&summary)?);
        } else {
            println!(
                "{}: {} candidates, {} approved, {} pending, {} text units",
                summary.action,
                summary.candidates,
                summary.approved,
                summary.pending,
                summary.units
            );
        }
    }
    Ok(())
}

fn collect_input(
    builder: &mut Builder,
    path: Option<&Path>,
    explicit: Option<FormatArg>,
    selector: &PathSelector,
    limit: usize,
) -> Result<()> {
    if path.is_some_and(|p| {
        p.extension().and_then(|e| e.to_str()).is_some_and(|e| {
            ["pdf", "docx", "xlsx", "pptx", "zip", "png", "jpg"]
                .iter()
                .any(|b| e.eq_ignore_ascii_case(b))
        })
    }) {
        bail!("terms discovery supports text, JSON and JSONL; extract binary documents first");
    }
    let source = match path {
        Some(p) => {
            let canonical = fs::canonicalize(p)
                .map_err(|_| anyhow::anyhow!("failed to resolve vocabulary input"))?;
            format!(
                "source:{}",
                terms::sha256(canonical.as_os_str().as_encoded_bytes())
            )
        }
        None => "stdin".to_owned(),
    };
    let (format, mut reader) = input::open(path, explicit)
        .map_err(|_| anyhow::anyhow!("failed to open vocabulary input"))?;
    if selector.is_restrictive() && format == FormatArg::Text {
        bail!("discovery selectors require JSON or JSONL");
    }
    if format == FormatArg::Json {
        let mut text = String::new();
        reader
            .take(limit as u64 + 1)
            .read_to_string(&mut text)
            .map_err(|_| anyhow::anyhow!("failed to read UTF-8 vocabulary input"))?;
        if text.len() > limit {
            bail!("JSON discovery document limit exceeded");
        }
        let value = parse_json(text.strip_prefix('\u{feff}').unwrap_or(&text), 1)?;
        walk(&value, "", "", &source, selector, builder, 0)?;
    } else {
        let mut number = 0;
        loop {
            let mut line = String::new();
            let read = reader
                .by_ref()
                .take(limit as u64 + 1)
                .read_line(&mut line)
                .map_err(|_| anyhow::anyhow!("failed to read UTF-8 vocabulary record"))?;
            if read == 0 {
                break;
            }
            number += 1;
            if line.len() > limit {
                bail!("vocabulary record limit exceeded at line {number}");
            }
            if format == FormatArg::Text {
                builder.ingest(&source, &format!("line[{number}]"), &line)?;
            } else if !line.trim().is_empty() {
                let text = if number == 1 {
                    line.strip_prefix('\u{feff}').unwrap_or(&line)
                } else {
                    &line
                };
                if text.trim().is_empty() {
                    continue;
                }
                let value = parse_json(text, number)?;
                walk(
                    &value,
                    "",
                    &format!("record[{number}]"),
                    &source,
                    selector,
                    builder,
                    0,
                )?;
            }
        }
    }
    Ok(())
}

fn parse_json(text: &str, number: usize) -> Result<Value> {
    serde_json::from_str(text)
        .map_err(|_| anyhow::anyhow!("invalid JSON vocabulary input at line {number}"))
}

fn walk(
    value: &Value,
    path: &str,
    prefix: &str,
    source: &str,
    selector: &PathSelector,
    builder: &mut Builder,
    depth: usize,
) -> Result<()> {
    if depth > 64 {
        bail!("vocabulary JSON depth limit exceeded");
    }
    match value {
        Value::String(text) if selector.should_scan(path) => {
            let location = if prefix.is_empty() {
                path.to_owned()
            } else if path.is_empty() {
                prefix.to_owned()
            } else {
                format!("{prefix}.{path}")
            };
            builder.ingest(source, &location, text)?;
        }
        Value::Array(items) => {
            for (i, child) in items.iter().enumerate() {
                walk(
                    child,
                    &format!("{path}[{i}]"),
                    prefix,
                    source,
                    selector,
                    builder,
                    depth + 1,
                )?;
            }
        }
        Value::Object(items) => {
            for (key, child) in items {
                let child_path = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                walk(
                    child,
                    &child_path,
                    prefix,
                    source,
                    selector,
                    builder,
                    depth + 1,
                )?;
            }
        }
        _ => (),
    }
    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let file = fs::File::open(path)
        .map_err(|_| anyhow::anyhow!("failed to read private vocabulary artifact"))?;
    let mut bytes = Vec::new();
    file.take(128 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| anyhow::anyhow!("failed to read private vocabulary artifact"))?;
    if bytes.len() > 128 * 1024 * 1024 {
        bail!("vocabulary artifact size limit exceeded");
    }
    serde_json::from_slice(&bytes).map_err(|_| anyhow::anyhow!("invalid vocabulary artifact JSON"))
}

fn write_json<T: Serialize>(path: &Path, value: &T, force: bool) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|_| anyhow::anyhow!("vocabulary serialization failed"))?;
    private_write(path, &bytes, force)
}

fn private_write(path: &Path, bytes: &[u8], force: bool) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut staged = tempfile::NamedTempFile::new_in(parent)
        .map_err(|_| anyhow::anyhow!("failed to stage private vocabulary artifact"))?;
    staged
        .write_all(bytes)
        .and_then(|()| staged.as_file().sync_all())
        .map_err(|_| anyhow::anyhow!("failed to write private vocabulary artifact"))?;
    if force {
        staged.persist(path)
    } else {
        staged.persist_noclobber(path)
    }
    .map_err(|_| {
        anyhow::anyhow!(
            "failed to publish vocabulary artifact; destination preserved (use --force to replace)"
        )
    })?;
    Ok(())
}
