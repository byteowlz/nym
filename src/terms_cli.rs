//! CLI and private artifact publication for model-independent vocabulary review.
use crate::{
    engine::selector::PathSelector,
    input::{self, FormatArg},
    terms::{self, Builder, Choice, Decision, Discovery, Review, Settings},
    terms_input::{FileStats, InputArgs},
    terms_progress::Scan,
};
use anyhow::{Result, bail};
use clap::{Args, Subcommand};
use serde::Serialize;
use serde_json::Value;
use std::{
    fs,
    io::{BufRead, IsTerminal, Read, Write},
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
    Discover(DiscoverArgs),
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

#[derive(Args)]
struct DiscoverArgs {
    #[command(flatten)]
    input: InputArgs,
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
    /// Bounded approximate candidate proposals, then exact full-corpus recount (named inputs only)
    #[arg(long)]
    bounded: bool,
    /// Show count-only progress on stderr (terminal bar; stage lines when redirected)
    #[arg(long, conflicts_with = "no_progress")]
    progress: bool,
    /// Disable progress, including interactive auto-progress
    #[arg(long)]
    no_progress: bool,
    /// Discovery-only replacement of unpaired UTF-16 value escapes; originals remain unchanged
    #[arg(long)]
    repair_unicode: bool,
    /// Prefer source-specific repetition over ubiquitous vocabulary; wider bounded recount pool
    #[arg(long, conflicts_with = "background")]
    specific: bool,
    #[arg(long)]
    phrase_words: Option<usize>,
    #[arg(long)]
    max_unit_bytes: Option<usize>,
    /// JSON map of public/reference literal frequency counts; ranking only
    #[arg(long)]
    background: Option<PathBuf>,
    #[arg(long)]
    force: bool,
}

#[derive(Serialize)]
struct Summary {
    action: &'static str,
    candidates: usize,
    approved: usize,
    units: u64,
    pending: usize,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    input_stats: Option<FileStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    selection: Option<terms::Selection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    projection: Option<crate::terms_unicode::Projection>,
}

pub fn run(common: &crate::CommonOpts, defaults: &Settings, command: TermsCommand) -> Result<()> {
    let summary = match command.action {
        Action::Discover(args) => discover(args, defaults, common.quiet)?,
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
                input_stats: None,
                selection: discovery.selection.clone(),
                projection: discovery.projection.clone(),
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
                input_stats: None,
                selection: discovery.selection.clone(),
                projection: discovery.projection.clone(),
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
            if let Some(projection) = &summary.projection {
                println!(
                    "Unicode projection: {} replacements in {} records; {} selected record units use view-relative offsets. Originals unchanged.",
                    projection.replacements, projection.records, projection.record_units
                );
            }
            if let Some(selection) = &summary.selection {
                println!(
                    "Bounded candidate proposals; {} literals recounted exactly. Omitted literals are unmeasured, not safe.",
                    selection.recounted
                );
            }
            if let Some(stats) = &summary.input_stats {
                println!(
                    "{} files; skipped {} extension-filtered, {} symlinks, {} artifacts",
                    stats.files,
                    stats.skipped_extensions,
                    stats.skipped_symlinks,
                    stats.skipped_artifacts
                );
            }
        }
    }
    Ok(())
}

fn discover(args: DiscoverArgs, defaults: &Settings, quiet: bool) -> Result<Summary> {
    let selector = PathSelector::new(&args.include, &args.exclude)
        .map_err(|_| anyhow::anyhow!("invalid discovery selector"))?;
    let mut excluded = vec![args.output.as_path()];
    if let Some(path) = &args.background {
        excluded.push(path.as_path());
    }
    let inputs = args.input.resolve(&excluded)?;
    let background = args.background.as_ref().map(|p| read_json(p)).transpose()?;
    let settings = Settings {
        max_distinct: args.max_distinct.unwrap_or(defaults.max_distinct),
        min_count: args.min_count.unwrap_or(defaults.min_count),
        max_candidates: args.limit.unwrap_or(defaults.max_candidates),
        max_phrase_words: args.phrase_words.unwrap_or(defaults.max_phrase_words),
        max_unit_bytes: args.max_unit_bytes.unwrap_or(defaults.max_unit_bytes),
        ..defaults.clone()
    };
    let limit = settings.max_unit_bytes;
    if args.bounded && (inputs.paths.is_empty() || background.is_some()) {
        bail!("bounded discovery requires named inputs and does not support background ranking");
    }
    let builder = if args.bounded {
        Builder::bounded(settings)?
    } else {
        Builder::new(settings, background)?
    };
    let discovery = args.collect(builder, &inputs.paths, &selector, limit, quiet)?;
    write_json(&args.output, &discovery, args.force)?;
    Ok(Summary {
        action: "discover",
        candidates: discovery.candidates.len(),
        approved: 0,
        units: discovery.unit_count,
        pending: discovery.candidates.len(),
        input_stats: Some(inputs.stats),
        selection: discovery.selection.clone(),
        projection: discovery.projection.clone(),
    })
}

impl DiscoverArgs {
    fn collect(
        &self,
        mut builder: Builder,
        paths: &[PathBuf],
        selector: &PathSelector,
        limit: usize,
        quiet: bool,
    ) -> Result<Discovery> {
        if self.repair_unicode {
            builder.enable_unicode_projection();
        }
        if self.specific {
            builder.prefer_specificity()?;
        }
        let enabled =
            !self.no_progress && !quiet && (self.progress || std::io::stderr().is_terminal());
        let stage = if self.bounded {
            "1/2 Proposal"
        } else {
            "Discovery"
        };
        collect_corpus(
            Scan::new(&mut builder, stage, paths.len(), enabled),
            paths,
            self.format,
            selector,
            limit,
        )?;
        if self.bounded {
            builder = builder.recount()?;
            collect_corpus(
                Scan::new(&mut builder, "2/2 Exact recount", paths.len(), enabled),
                paths,
                self.format,
                selector,
                limit,
            )?;
        }
        builder.finish()
    }
}

fn collect_corpus(
    mut scan: Scan<'_>,
    paths: &[PathBuf],
    format: Option<FormatArg>,
    selector: &PathSelector,
    limit: usize,
) -> Result<()> {
    if paths.is_empty() {
        collect_input(&mut scan, None, format, selector, limit)?;
        scan.file_scanned();
    }
    for path in paths {
        collect_input(&mut scan, Some(path), format, selector, limit)?;
        scan.file_scanned();
    }
    scan.finish();
    Ok(())
}

fn collect_input(
    builder: &mut Scan<'_>,
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
    builder.end_record();
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
        let value = builder.parse_record(&source, "document", &text, 1)?;
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
                let value =
                    builder.parse_record(&source, &format!("record[{number}]"), &line, number)?;
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

fn walk(
    value: &Value,
    path: &str,
    prefix: &str,
    source: &str,
    selector: &PathSelector,
    builder: &mut Scan<'_>,
    depth: usize,
) -> Result<()> {
    // Match serde_json's default recursion budget; never disable its parser guard.
    // Directory traversal retains its separate 64-level budget.
    if depth > 128 {
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
