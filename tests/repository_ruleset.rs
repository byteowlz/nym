//! The `repository` ruleset: credentials and personal data in a source tree
//! are found, while lockfile checksums, package names, ids, references and
//! named fixtures stay quiet. Synthetic values only.
use serde_json::Value;
use std::{fs, process::Command};

fn detect(text: &str) -> Vec<(String, String)> {
    let dir = tempfile::tempdir().unwrap();
    let cfg = dir.path().join("config.toml");
    fs::write(&cfg, "").unwrap();
    let input = dir.path().join("repo.txt");
    fs::write(&input, text).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nym"))
        .arg("--config")
        .arg(&cfg)
        .args(["detect", "--json", "--ruleset", "repository"])
        .arg(&input)
        .env("XDG_CONFIG_HOME", dir.path().join("xdg"))
        .env("HF_HUB_OFFLINE", "1")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let findings: Value = serde_json::from_slice(&out.stdout).unwrap();
    findings
        .as_array()
        .unwrap()
        .iter()
        .map(|f| {
            (
                f["pattern_name"].as_str().unwrap().to_owned(),
                f["matched_text"].as_str().unwrap().to_owned(),
            )
        })
        .collect()
}

#[test]
fn lockfiles_ids_references_and_fixtures_are_quiet() {
    let noise = r#"
[[package]]
name = "serde"
version = "1.0.228"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "9a8e94ea7f378bd32cbbd37198a4a91436180c5bb472411e48b5ec2e2124ae9e"
"@types/node": { "resolved": "https://registry.npmjs.org/@types/node/-/node-22.7.4.tgz", "integrity": "sha512-y+NPi1rFzDs1NdQHHToqeiX2TIS79SWEAw9GYhkkx8bsBVBsfrdYQZqMFNM5phyEUXNj0Px0fT7SnEMpmRxOg==" }
request_id = "3f2a9c1e-7b4d-4e8a-9f1c-2d6b8e0a5c37"
api_key = "env:OPENAI_API_KEY"
token = "keychain:eavs-openai"
password = "${DB_PASSWORD}"
API_TOKEN=synthetic-access
let token = "your-token-here";
see https://github.com/byteowlz/eavs and run on 10.0.0.12 or 192.168.1.4
DATABASE_URL=postgres://postgres:postgres@localhost:5432/app
git clone git@github.com:byteowlz/eavs.git
"#;
    let found = detect(noise);
    assert!(found.is_empty(), "unexpected findings: {found:?}");
}

#[test]
fn real_credentials_and_personal_data_are_found() {
    let leaks = r#"
OPENAI_API_KEY=sk-proj-Kq7vR2xLp9WmT4nZsE6uYc8HdF3gJ5aBiN1oVr0tQx
AWS_ACCESS_KEY_ID=AKIAQWERTYUIOP123456
DB_PASSWORD=correct-horse-battery
DATABASE_URL=postgres://app:Kq7vR2xLp9@db.internal:5432/orders
maintainer: quillon.example@posteo.de
let config = "/Users/quillon/projects/app/config.toml";
-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEAsyntheticsyntheticsynthetic
-----END RSA PRIVATE KEY-----
"#;
    let found = detect(leaks);
    let kinds: Vec<&str> = found.iter().map(|(k, _)| k.as_str()).collect();
    for kind in [
        "vendor_secret",
        "aws_key",
        "secret_assignment",
        "url_credentials",
        "email",
        "home_dir",
        "private_key",
    ] {
        assert!(kinds.contains(&kind), "{kind} missing from {found:?}");
    }
    let url = found.iter().find(|(k, _)| k == "url_credentials").unwrap();
    assert_eq!(url.1, "Kq7vR2xLp9", "only the URL password is the finding");
    assert!(
        found
            .iter()
            .any(|(k, v)| k == "secret_assignment" && v == "correct-horse-battery")
    );
}
