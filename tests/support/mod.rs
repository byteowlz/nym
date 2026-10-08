use serde_json::Value;
use std::{fs, path::Path};

/// Load and sanity-check the synthetic public discovery artifact shared by CLI tests.
pub fn discovery_value(path: &Path) -> Value {
    let value: Value = serde_json::from_reader(fs::File::open(path).unwrap()).unwrap();
    assert_eq!(value["schema"], "nym.terms.discovery.v1");
    value
}
