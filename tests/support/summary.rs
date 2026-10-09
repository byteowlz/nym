/// Parse a machine response only after proving its expected exit status.
pub fn summary(output: std::process::Output, status: i32) -> serde_json::Value {
    assert_eq!(
        output.status.code(),
        Some(status),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
