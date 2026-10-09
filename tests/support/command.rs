use std::{
    fs,
    process::{Command, Output},
};
use tempfile::TempDir;

/// Build a CLI command using an isolated synthetic config and XDG roots.
pub fn nym_command(home: &TempDir) -> Command {
    let config = home.path().join("config.toml");
    if !config.exists() {
        fs::write(&config, "").unwrap();
    }
    let mut command = Command::new(env!("CARGO_BIN_EXE_nym"));
    command.args(["--config", config.to_str().unwrap()]);
    command
        .env("HOME", home.path())
        .env("XDG_STATE_HOME", home.path().join("state"))
        .env("XDG_DATA_HOME", home.path().join("data"));
    command
}

pub fn run(home: &TempDir, args: &[&str]) -> Output {
    nym_command(home).args(args).output().unwrap()
}
