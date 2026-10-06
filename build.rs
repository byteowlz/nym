use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn main() {
    // Leave Cargo's default package-wide change detection enabled. Watching only
    // HEAD would incorrectly retain clean provenance after source edits.
    let revision = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = git(&["status", "--porcelain", "--untracked-files=normal"])
        .map(|status| !status.is_empty());
    let mut features: Vec<_> = std::env::vars()
        .filter_map(|(key, _)| {
            key.strip_prefix("CARGO_FEATURE_")
                .map(|feature| feature.to_ascii_lowercase().replace('_', "-"))
        })
        .collect();
    features.sort();
    let features = if features.is_empty() {
        "minimal".into()
    } else {
        features.join(",")
    };
    let state = match dirty {
        Some(true) => "dirty",
        Some(false) => "clean",
        None => "unknown",
    };
    let target = std::env::var("TARGET").expect("Cargo supplies TARGET");
    let version = std::env::var("CARGO_PKG_VERSION").expect("Cargo supplies package version");
    println!(
        "cargo:rustc-env=NYM_BUILD_VERSION={version}+g{revision}.{state} (features={features};target={target})"
    );
}
