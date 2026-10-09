//! Private saved-run inventory; metadata never contains corpus text or serving capabilities.
use crate::{
    config::{self, Config},
    terms::{Discovery, Review},
    terms_cli::private_write,
};
use anyhow::{Result, bail};
use clap::{Args, Subcommand};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::UNIX_EPOCH,
};

const SCHEMA: &str = "nym.terms.run.v1";
const MAX_RUNS: usize = 10_000;
const MAX_ARTIFACT: u64 = 128 * 1024 * 1024;

#[derive(Args)]
pub struct RunsCommand {
    #[command(subcommand)]
    action: Action,
}

#[derive(Subcommand)]
enum Action {
    /// List saved runs and legacy Pi-run directories, without reading contexts
    List,
    /// Show run metadata; ID may be a unique prefix or 'latest'
    Show {
        #[arg(default_value = "latest")]
        id: String,
    },
    /// Register a completed discovery.json + review.html directory; no source scan
    Register {
        directory: PathBuf,
        #[arg(long)]
        force: bool,
    },
    /// Open a verified saved review in the desktop browser
    Open {
        #[arg(default_value = "latest")]
        id: String,
    },
    /// Share a verified run; foreground, capability-gated and read-only
    Serve(ServeArgs),
}

#[derive(Args)]
struct ServeArgs {
    #[arg(default_value = "latest")]
    id: String,
    /// Bind to the Tailscale IPv4 address instead of loopback
    #[arg(long)]
    tailnet: bool,
    #[arg(long, default_value_t = 8521)]
    port: u16,
    #[arg(long)]
    open: bool,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Run {
    schema: String,
    id: String,
    directory: PathBuf,
    created_unix_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    candidates: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    units: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    discovery_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    html_sha256: Option<String>,
}

#[derive(Serialize)]
struct Entry {
    #[serde(flatten)]
    run: Run,
    registered: bool,
    status: &'static str,
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn run_id(directory: &Path) -> Result<String> {
    Ok(digest(
        directory
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("run path must be valid Unicode"))?
            .as_bytes(),
    ))
}
fn root(configured: &Option<String>, fallback: fn() -> PathBuf) -> Result<PathBuf> {
    let path = configured
        .as_ref()
        .map_or_else(fallback, |value| config::expand_path(value));
    if !path.is_absolute() {
        bail!("run storage root must be absolute");
    }
    Ok(path)
}
fn registry(config: &Config) -> Result<PathBuf> {
    Ok(root(&config.paths.state_dir, config::state_dir)?.join("terms/runs"))
}
fn private_directory(path: &Path) -> Result<()> {
    if path.is_symlink() {
        bail!("run storage cannot be a symlink");
    }
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder
        .create(path)
        .map_err(|_| anyhow::anyhow!("failed to create private run storage"))?;
    Ok(())
}
fn read_regular(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let meta =
        fs::symlink_metadata(path).map_err(|_| anyhow::anyhow!("saved run file unavailable"))?;
    if !meta.is_file() || meta.len() > limit {
        bail!("saved run requires bounded regular files, not links");
    }
    let file = fs::File::open(path).map_err(|_| anyhow::anyhow!("saved run file unavailable"))?;
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| anyhow::anyhow!("failed to read saved run"))?;
    if bytes.len() as u64 > limit {
        bail!("saved run file size limit exceeded");
    }
    Ok(bytes)
}
fn directory(path: &Path) -> Result<PathBuf> {
    let meta = fs::symlink_metadata(path)
        .map_err(|_| anyhow::anyhow!("saved run directory unavailable"))?;
    if !meta.is_dir() {
        bail!("saved run must be a real directory, not a link");
    }
    path.canonicalize()
        .map_err(|_| anyhow::anyhow!("saved run directory unavailable"))
}
fn legacy(path: &Path) -> Result<Run> {
    let directory = directory(path)?;
    let metadata =
        fs::metadata(&directory).map_err(|_| anyhow::anyhow!("saved run metadata unavailable"))?;
    let created = metadata
        .created()
        .or_else(|_| metadata.modified())
        .map_err(|_| anyhow::anyhow!("saved run time unavailable"))?;
    let created_unix_ms = u64::try_from(
        created
            .duration_since(UNIX_EPOCH)
            .map_err(|_| anyhow::anyhow!("invalid saved run time"))?
            .as_millis(),
    )?;
    Ok(Run {
        schema: SCHEMA.into(),
        id: run_id(&directory)?,
        directory,
        created_unix_ms,
        candidates: None,
        units: None,
        discovery_sha256: None,
        html_sha256: None,
    })
}
fn valid_run(run: &Run) -> Result<()> {
    if run.schema != SCHEMA
        || !run.directory.is_absolute()
        || run.id != run_id(&run.directory)?
        || run.id.len() != 64
        || !run.id.bytes().all(|c| c.is_ascii_hexdigit())
        || run.candidates.is_some_and(|count| count > 5000)
        || run
            .html_sha256
            .as_ref()
            .is_some_and(|s| s.len() != 64 || !s.bytes().all(|c| c.is_ascii_hexdigit()))
        || run
            .discovery_sha256
            .as_ref()
            .is_some_and(|s| s.len() != 64 || !s.bytes().all(|c| c.is_ascii_hexdigit()))
        || [
            run.candidates.is_some(),
            run.units.is_some(),
            run.discovery_sha256.is_some(),
            run.html_sha256.is_some(),
        ]
        .iter()
        .any(|v| !v)
    {
        bail!("invalid saved run manifest; inventory incomplete");
    }
    Ok(())
}
fn entries(path: &Path) -> Result<Vec<fs::DirEntry>> {
    if !path.exists() && !path.is_symlink() {
        return Ok(Vec::new());
    }
    directory(path)?;
    let mut entries = Vec::new();
    for entry in fs::read_dir(path).map_err(|_| anyhow::anyhow!("run inventory unreadable"))? {
        entries.push(entry.map_err(|_| anyhow::anyhow!("run inventory unreadable"))?);
        if entries.len() > MAX_RUNS {
            bail!("saved run inventory limit exceeded");
        }
    }
    Ok(entries)
}
fn inventory(config: &Config) -> Result<Vec<Entry>> {
    let mut runs = std::collections::BTreeMap::new();
    for file in entries(&registry(config)?)? {
        let path = file.path();
        let run: Run = serde_json::from_slice(&read_regular(&path, 64 * 1024)?)
            .map_err(|_| anyhow::anyhow!("invalid saved run manifest; inventory incomplete"))?;
        valid_run(&run)?;
        if path.file_stem().and_then(|s| s.to_str()) != Some(run.id.as_str()) {
            bail!("saved run manifest identity mismatch");
        }
        runs.insert(run.id.clone(), (run, true));
    }
    let legacy_root = root(&config.paths.state_dir, config::state_dir)?.join("pi-terms");
    for entry in entries(&legacy_root)? {
        if !entry
            .file_type()
            .map_err(|_| anyhow::anyhow!("run inventory unreadable"))?
            .is_dir()
        {
            if entry.path().is_symlink() {
                bail!("saved run inventory contains a link");
            }
            continue;
        }
        let run = legacy(&entry.path())?;
        runs.entry(run.id.clone()).or_insert((run, false));
    }
    if runs.len() > MAX_RUNS {
        bail!("saved run inventory limit exceeded");
    }
    let mut result: Vec<_> = runs
        .into_values()
        .map(|(run, registered)| {
            let status = if run.directory.is_symlink() {
                "unavailable"
            } else if !run.directory.is_dir() {
                "missing"
            } else if ["discovery.json", "review.html"].iter().all(|name| {
                fs::symlink_metadata(run.directory.join(name)).is_ok_and(|m| m.is_file())
            }) {
                "ready"
            } else {
                "incomplete"
            };
            Entry {
                run,
                registered,
                status,
            }
        })
        .collect();
    result.sort_by(|a, b| {
        b.run
            .created_unix_ms
            .cmp(&a.run.created_unix_ms)
            .then_with(|| a.run.id.cmp(&b.run.id))
    });
    Ok(result)
}
fn select(mut entries: Vec<Entry>, id: &str) -> Result<Entry> {
    if id == "latest" {
        return entries
            .into_iter()
            .find(|e| e.status == "ready")
            .ok_or_else(|| anyhow::anyhow!("no completed saved review run"));
    }
    if id.len() < 4 || id.len() > 64 || !id.bytes().all(|c| c.is_ascii_hexdigit()) {
        bail!("run must be 'latest' or a unique hexadecimal ID prefix (at least 4 characters)");
    }
    entries.retain(|entry| entry.run.id.starts_with(id));
    if entries.len() != 1 {
        bail!("run ID is missing or ambiguous; list saved runs");
    }
    Ok(entries.remove(0))
}
fn verified(run: &Run) -> Result<(Discovery, Review, String)> {
    directory(&run.directory)?;
    let discovery: Discovery = serde_json::from_slice(&read_regular(
        &run.directory.join("discovery.json"),
        MAX_ARTIFACT,
    )?)
    .map_err(|_| anyhow::anyhow!("invalid saved discovery"))?;
    discovery.validate()?;
    let html = read_regular(&run.directory.join("review.html"), MAX_ARTIFACT)?;
    let document =
        std::str::from_utf8(&html).map_err(|_| anyhow::anyhow!("invalid saved review UTF-8"))?;
    let marker = "<script type=\"application/json\" id=\"data\">";
    if document.matches(marker).count() != 1 {
        bail!("invalid saved review data block");
    }
    let (_, rest) = document
        .split_once(marker)
        .ok_or_else(|| anyhow::anyhow!("invalid saved review data"))?;
    let (raw, _) = rest
        .split_once("</script>")
        .ok_or_else(|| anyhow::anyhow!("invalid saved review data"))?;
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Data {
        discovery: Discovery,
        review: Review,
    }
    let data: Data =
        serde_json::from_str(raw).map_err(|_| anyhow::anyhow!("invalid saved review data"))?;
    data.discovery.validate()?;
    data.review.validate(&discovery)?;
    let fingerprint = discovery.fingerprint()?;
    let html_sha256 = digest(&html);
    if data.discovery.fingerprint()? != fingerprint
        || run
            .discovery_sha256
            .as_ref()
            .is_some_and(|hash| hash != &fingerprint)
        || run
            .html_sha256
            .as_ref()
            .is_some_and(|hash| hash != &html_sha256)
    {
        bail!("saved run changed or review scope mismatched; re-register explicitly with --force");
    }
    Ok((discovery, data.review, html_sha256))
}
fn register(config: &Config, path: &Path, force: bool) -> Result<Entry> {
    let mut run = legacy(path)?;
    let (discovery, _, hash) = verified(&run)?;
    run.candidates = Some(discovery.candidates.len());
    run.units = Some(discovery.unit_count);
    run.discovery_sha256 = Some(discovery.fingerprint()?);
    run.html_sha256 = Some(hash);
    let registry = registry(config)?;
    private_directory(&registry)?;
    let target = registry.join(format!("{}.json", run.id));
    if target.exists() || target.is_symlink() {
        let old: Run = serde_json::from_slice(&read_regular(&target, 64 * 1024)?)
            .map_err(|_| anyhow::anyhow!("invalid existing run manifest; preserved"))?;
        valid_run(&old)?;
        if old.directory != run.directory {
            bail!("saved run identity collision; manifest preserved");
        }
        run.created_unix_ms = old.created_unix_ms;
        if old.discovery_sha256 == run.discovery_sha256 && old.html_sha256 == run.html_sha256 {
            return Ok(Entry {
                run: old,
                registered: true,
                status: "ready",
            });
        }
        if !force {
            bail!("run already registered with different content; use --force explicitly");
        }
    }
    if let Err(error) = private_write(&target, &serde_json::to_vec_pretty(&run)?, force) {
        // An identical concurrent registration can win no-clobber publication.
        let winner: Run =
            serde_json::from_slice(&read_regular(&target, 64 * 1024)?).map_err(|_| {
                anyhow::anyhow!("run registration publication failed; inventory preserved")
            })?;
        valid_run(&winner)?;
        if force
            || winner.directory != run.directory
            || winner.discovery_sha256 != run.discovery_sha256
            || winner.html_sha256 != run.html_sha256
        {
            return Err(error);
        }
        return Ok(Entry {
            run: winner,
            registered: true,
            status: "ready",
        });
    }
    Ok(Entry {
        run,
        registered: true,
        status: "ready",
    })
}
fn emit<T: Serialize>(common: &crate::CommonOpts, value: &T, text: impl FnOnce()) -> Result<()> {
    if !common.quiet {
        if common.json {
            println!("{}", serde_json::to_string(value)?);
        } else if common.yaml {
            print!("{}", serde_yaml::to_string(value)?);
        } else {
            text();
        }
    }
    Ok(())
}
fn show(entry: &Entry) {
    println!(
        "{} · {} · {}\nDirectory: {}",
        entry.run.id,
        entry.status,
        if entry.registered {
            "registered"
        } else {
            "legacy"
        },
        entry.run.directory.display()
    );
    if let (Some(candidates), Some(units)) = (entry.run.candidates, entry.run.units) {
        println!("{candidates} candidates · {units} text units");
    }
}
fn open(config: &Config, run: &Run) -> Result<()> {
    let (discovery, review, _) = verified(run)?;
    // Never execute supplied HTML/scripts. Open a trusted, private rendered snapshot.
    let views = root(&config.paths.state_dir, config::state_dir)?.join("terms/views");
    private_directory(&views)?;
    let page = views.join(format!("{}.html", run.id));
    private_write(
        &page,
        crate::terms_cli::render_html(&discovery, &review)?.as_bytes(),
        true,
    )?;
    #[cfg(target_os = "macos")]
    let program = "open";
    #[cfg(target_os = "windows")]
    let program = "explorer.exe";
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let program = "xdg-open";
    let status = Command::new(program)
        .arg(page)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|_| anyhow::anyhow!("review browser opener unavailable"))?;
    if !status.success() {
        bail!("review browser could not open; use runs show for its location");
    }
    Ok(())
}
fn host_assets(config: &Config) -> Result<PathBuf> {
    let assets = [
        (
            "scripts/serve_terms_review.py",
            include_str!("../scripts/serve_terms_review.py"),
        ),
        (
            "scripts/json_safe.py",
            include_str!("../scripts/json_safe.py"),
        ),
        ("src/terms_review.html", include_str!("terms_review.html")),
    ];
    let mut hasher = Sha256::new();
    for (name, body) in assets {
        hasher.update(name.as_bytes());
        hasher.update([0]);
        hasher.update(body.as_bytes());
    }
    let version = format!("{:x}", hasher.finalize());
    let root = root(&config.paths.data_dir, config::data_dir)?
        .join("terms/review-host")
        .join(version);
    for (name, body) in assets {
        let path = root.join(name);
        private_directory(
            path.parent()
                .ok_or_else(|| anyhow::anyhow!("invalid review host storage"))?,
        )?;
        if !path.exists() && !path.is_symlink() {
            // Racing identical installations may win publication; verify the winner below.
            let _ = private_write(&path, body.as_bytes(), false);
        }
        if read_regular(&path, MAX_ARTIFACT)? != body.as_bytes() {
            bail!("review host asset integrity failure; no server started");
        }
    }
    Ok(root.join("scripts/serve_terms_review.py"))
}
fn serve(common: &crate::CommonOpts, config: &Config, run: &Run, args: &ServeArgs) -> Result<()> {
    let (_, _, hash) = verified(run)?;
    let helper = host_assets(config)?;
    let mut command = Command::new("uv");
    command
        .args(["run", "--no-project", "python"])
        .arg(helper)
        .arg(run.directory.join("review.html"));
    command
        .arg("--expected-sha256")
        .arg(hash)
        .arg("--port")
        .arg(args.port.to_string());
    if args.tailnet {
        command.arg("--tailnet");
    }
    if args.open {
        command.arg("--open");
    }
    if common.json {
        command.arg("--json");
    }
    if common.yaml {
        command.arg("--yaml");
    }
    if common.quiet {
        command.stdout(Stdio::null());
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // Replace the CLI so Ctrl-C owns the entire host lifecycle; no private staging files.
        let _ = command.exec();
        bail!("review hosting requires uv; failed to start private host");
    }
    #[cfg(not(unix))]
    {
        let status = command
            .status()
            .map_err(|_| anyhow::anyhow!("review hosting requires uv"))?;
        if !status.success() {
            bail!("private review host failed");
        }
        Ok(())
    }
}

pub fn run(common: &crate::CommonOpts, config: &Config, command: RunsCommand) -> Result<()> {
    match command.action {
        Action::List => {
            let entries = inventory(config)?;
            emit(common, &entries, || {
                if entries.is_empty() {
                    println!("No saved vocabulary runs.");
                }
                for entry in &entries {
                    println!(
                        "{}  {}  {}  {}",
                        &entry.run.id[..12],
                        i64::try_from(entry.run.created_unix_ms)
                            .ok()
                            .and_then(chrono::DateTime::from_timestamp_millis)
                            .map_or_else(
                                || "unknown-time".to_string(),
                                |time| time.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
                            ),
                        entry.status,
                        if entry.registered {
                            "registered"
                        } else {
                            "legacy"
                        }
                    );
                }
            })
        }
        Action::Register { directory, force } => {
            let entry = register(config, &directory, force)?;
            emit(common, &entry, || show(&entry))
        }
        Action::Show { id } => {
            let entry = select(inventory(config)?, &id)?;
            emit(common, &entry, || show(&entry))
        }
        Action::Open { id } => {
            let entry = select(inventory(config)?, &id)?;
            open(config, &entry.run)?;
            emit(common, &entry, || show(&entry))
        }
        Action::Serve(args) => {
            let entry = select(inventory(config)?, &args.id)?;
            serve(common, config, &entry.run, &args)
        }
    }
}
