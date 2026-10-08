//! Resolve a bounded, deterministic file set before vocabulary extraction.
use anyhow::{Result, bail};
use clap::Args;
use serde::Serialize;
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

const MAX_FILES: usize = 10_000;
const MAX_ENTRIES: usize = 100_000;
const MAX_DEPTH: usize = 64;
const DEFAULT_EXTENSIONS: &[&str] = &["txt", "text", "log", "md", "json", "jsonl", "ndjson"];

#[derive(Args)]
pub(crate) struct InputArgs {
    /// Input files or directory roots; omit to read stdin
    pub inputs: Vec<PathBuf>,
    /// Descend directory roots, including hidden entries; never follow symlinks
    #[arg(short = 'r', long)]
    pub recursive: bool,
    /// Directory file extensions, repeatable/comma-separated (default: txt,text,log,md,json,jsonl,ndjson)
    #[arg(long = "extension", value_delimiter = ',')]
    pub extensions: Vec<String>,
    /// Additional files to exclude from traversal, by exact path (not JSON selectors)
    #[arg(long = "exclude-file")]
    pub exclude_files: Vec<PathBuf>,
    /// Maximum unique selected files (1..=10000); errors never publish partial discovery
    #[arg(long, default_value_t = MAX_FILES, value_parser = clap::value_parser!(usize))]
    pub max_files: usize,
}

#[derive(Default, Serialize)]
pub(crate) struct FileStats {
    pub files: usize,
    pub skipped_extensions: usize,
    pub skipped_symlinks: usize,
    pub skipped_artifacts: usize,
}

pub(crate) struct FileSet {
    pub paths: Vec<PathBuf>,
    pub stats: FileStats,
}

impl InputArgs {
    pub fn resolve(&self, excluded: &[&Path]) -> Result<FileSet> {
        if self.max_files == 0 || self.max_files > MAX_FILES {
            bail!("discovery --max-files must be between 1 and 10000");
        }
        if self.inputs.is_empty()
            && (self.recursive || !self.extensions.is_empty() || !self.exclude_files.is_empty())
        {
            bail!("directory traversal options require named inputs");
        }
        let extensions = if self.extensions.is_empty() {
            DEFAULT_EXTENSIONS
                .iter()
                .map(|value| (*value).to_owned())
                .collect()
        } else {
            self.extensions.iter().map(|value| {
                let value = value.strip_prefix('.').unwrap_or(value);
                if value.is_empty() || value.len() > 16 || !value.bytes().all(|b| b.is_ascii_alphanumeric()) {
                    bail!("discovery extensions must be simple extension names, not globs or paths");
                }
                Ok(value.to_ascii_lowercase())
            }).collect::<Result<BTreeSet<_>>>()?
        };
        let mut walker = Walker {
            recursive: self.recursive,
            max_files: self.max_files,
            extensions,
            excluded: excluded
                .iter()
                .copied()
                .chain(self.exclude_files.iter().map(PathBuf::as_path))
                .map(destination_identity)
                .collect::<Result<_>>()?,
            files: BTreeSet::new(),
            directories: BTreeSet::new(),
            entries: 0,
            stats: FileStats::default(),
        };
        for path in &self.inputs {
            walker.visit(path, 0, true)?;
        }
        if !self.inputs.is_empty() && walker.files.is_empty() {
            bail!("no supported vocabulary input files found");
        }
        walker.stats.files = walker.files.len();
        Ok(FileSet {
            paths: walker.files.into_iter().collect(),
            stats: walker.stats,
        })
    }
}

struct Walker {
    recursive: bool,
    max_files: usize,
    extensions: BTreeSet<String>,
    excluded: BTreeSet<PathBuf>,
    files: BTreeSet<PathBuf>,
    directories: BTreeSet<PathBuf>,
    entries: usize,
    stats: FileStats,
}

impl Walker {
    fn visit(&mut self, path: &Path, depth: usize, explicit: bool) -> Result<()> {
        let metadata = fs::symlink_metadata(path)
            .map_err(|_| anyhow::anyhow!("failed to inspect vocabulary input tree"))?;
        if metadata.file_type().is_symlink() {
            if explicit {
                bail!("explicit vocabulary input symlinks are not supported; use the real path");
            }
            self.stats.skipped_symlinks += 1;
            return Ok(());
        }
        let canonical = fs::canonicalize(path)
            .map_err(|_| anyhow::anyhow!("failed to resolve vocabulary input tree"))?;
        if self.excluded.contains(&canonical) {
            if explicit {
                bail!("vocabulary artifact cannot also be an explicit input");
            }
            self.stats.skipped_artifacts += 1;
            return Ok(());
        }
        if metadata.is_dir() {
            return self.directory(&canonical, depth);
        }
        if !metadata.is_file() {
            if explicit {
                bail!("vocabulary input must be a regular file or directory");
            }
            // FIFOs/devices/sockets must never be opened during discovery.
            bail!("unsupported nonregular entry in vocabulary input tree");
        }
        if !explicit
            && !path
                .extension()
                .and_then(|value| value.to_str())
                .is_some_and(|value| self.extensions.contains(&value.to_ascii_lowercase()))
        {
            self.stats.skipped_extensions += 1;
            return Ok(());
        }
        self.files.insert(canonical);
        if self.files.len() > self.max_files {
            bail!("vocabulary file limit exceeded; narrow the input roots or adjust --max-files");
        }
        Ok(())
    }

    fn directory(&mut self, path: &Path, depth: usize) -> Result<()> {
        if !self.recursive {
            bail!("directory vocabulary inputs require --recursive");
        }
        if depth > MAX_DEPTH {
            bail!("vocabulary directory depth limit exceeded");
        }
        if !self.directories.insert(path.to_owned()) {
            return Ok(());
        }
        let entries = fs::read_dir(path)
            .map_err(|_| anyhow::anyhow!("failed to read vocabulary input directory"))?;
        let mut children = Vec::new();
        for entry in entries {
            self.entries += 1;
            if self.entries > MAX_ENTRIES {
                bail!("vocabulary traversal entry limit exceeded");
            }
            children.push(
                entry
                    .map_err(|_| anyhow::anyhow!("failed to read vocabulary directory entry"))?
                    .path(),
            );
        }
        children.sort();
        for child in children {
            self.visit(&child, depth + 1, false)?;
        }
        Ok(())
    }
}

fn destination_identity(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        return fs::canonicalize(path)
            .map_err(|_| anyhow::anyhow!("failed to resolve vocabulary artifact destination"));
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("invalid vocabulary artifact destination"))?;
    Ok(fs::canonicalize(parent)
        .map_err(|_| anyhow::anyhow!("vocabulary artifact parent directory must exist"))?
        .join(name))
}
