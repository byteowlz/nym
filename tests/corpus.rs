//! Synthetic-only corpus accounting and transformation contract.
use serde_json::{Value, json};
#[path = "support/summary.rs"]
mod summary;
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};
fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_nym"))
        .args(args)
        .output()
        .unwrap()
}
fn ok(out: Output) -> Value {
    summary::summary(out, 0)
}
fn snapshot(root: &Path, plan: &Path, scope: &str) -> Value {
    ok(run(&[
        "corpus",
        "inventory",
        root.to_str().unwrap(),
        "-o",
        plan.to_str().unwrap(),
        "--scope",
        scope,
    ]));
    serde_json::from_slice(&fs::read(plan).unwrap()).unwrap()
}
fn reviewed(plan: &mut Value) {
    for source in plan["sources"].as_array_mut().unwrap() {
        source["review"] = json!({"basis":"synthetic","evidence":"owned synthetic fixture","privacy_review":"fixture review only","serialization_reviewed":true,"media_reviewed":true,"publish_verbatim":true});
        for unit in source["units"].as_array_mut().unwrap() {
            unit["approval"] = "fixture".into();
            unit["action"] = json!({"kind":"retain"});
        }
    }
}
fn save(path: &Path, value: &Value) {
    fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
}
fn applied(plan: &Path, root: &Path, out: &Path, ledger: &Path) -> Output {
    run(&[
        "corpus",
        "apply",
        plan.to_str().unwrap(),
        "--source",
        root.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
        "--ledger",
        ledger.to_str().unwrap(),
    ])
}
#[test]
fn artifact_schema_is_inspectable_from_the_binary() {
    let schema = ok(run(&["corpus", "schema", "--json"]));
    assert_eq!(
        schema,
        serde_json::from_str::<Value>(include_str!("../examples/corpus.schema.json")).unwrap()
    );
    assert_eq!(schema["$defs"]["plan"]["properties"]["version"]["const"], 1);
}

#[test]
fn corpus_accounts_keys_numbers_structural_fields_and_opaque_assets() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("input");
    fs::create_dir(&root).unwrap();
    fs::write(root.join("trace.jsonl"),"{\"role\":\"assistant\",\"owner@example.invalid\":9007199254740993123456789,\"empty\":{},\"items\":[],\"flag\":false,\"nil\":null}\n\n").unwrap();
    fs::write(root.join("opaque.bin"), [0, 1, 255]).unwrap();
    let path = dir.path().join("plan.json");
    let plan = snapshot(&root, &path, "session");
    let mut kinds = std::collections::BTreeSet::new();
    for source in plan["sources"].as_array().unwrap() {
        for unit in source["units"].as_array().unwrap() {
            kinds.insert(unit["kind"].as_str().unwrap());
            assert_eq!(unit["action"]["kind"], "unresolved");
        }
    }
    assert_eq!(
        kinds,
        std::collections::BTreeSet::from([
            "asset",
            "boolean",
            "empty_array",
            "empty_object",
            "filename",
            "key",
            "layout",
            "null",
            "number",
            "string"
        ])
    );
    let blocked = run(&[
        "corpus",
        "validate",
        path.to_str().unwrap(),
        "--source",
        root.to_str().unwrap(),
        "--json",
    ]);
    assert!(!blocked.status.success());
    let blocked_json: Value = serde_json::from_slice(&blocked.stdout).unwrap();
    assert_eq!(blocked_json["error"]["code"], "corpus_blocked");
    assert_eq!(blocked_json["error"]["state"], "blocked");
    assert!(
        blocked_json["error"]["reason"]
            .as_str()
            .is_some_and(|reason| !reason.is_empty()),
        "operators need the value-free reason: {blocked_json}"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
}
#[test]
fn full_and_session_scopes_are_explicit_and_deterministic() {
    for scope in ["full", "session"] {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("input");
        fs::create_dir(&root).unwrap();
        fs::write(root.join("a.json"),r#"{"role":"user","text":"Alice","number":9007199254740993123456789,"flag":false,"empty":[],"nil":null}"#).unwrap();
        fs::write(root.join("b.json"), r#"{"role":"tool","text":"Alice"}"#).unwrap();
        let plan_path = dir.path().join("plan.json");
        let mut plan = snapshot(&root, &plan_path, scope);
        reviewed(&mut plan);
        for source in plan["sources"].as_array_mut().unwrap() {
            for unit in source["units"].as_array_mut().unwrap() {
                if unit["original"] == "Alice" {
                    unit["action"] = json!({"kind":"fake","pattern":"first_name"});
                }
            }
        }
        save(&plan_path, &plan);
        let out = dir.path().join("out");
        let ledger = dir.path().join("ledger.json");
        ok(applied(&plan_path, &root, &out, &ledger));
        let a: Value =
            serde_json::from_slice(&fs::read(out.join("sources/a.json")).unwrap()).unwrap();
        let b: Value =
            serde_json::from_slice(&fs::read(out.join("sources/b.json")).unwrap()).unwrap();
        if scope == "full" {
            assert_eq!(a["text"], b["text"]);
        } else {
            assert_ne!(a["text"], b["text"]);
        }
        assert_eq!(a["role"], "user");
        assert_eq!(b["role"], "tool");
        assert!(
            fs::read_to_string(out.join("sources/a.json"))
                .unwrap()
                .contains("9007199254740993123456789")
        );
        let second = dir.path().join("second");
        let second_ledger = dir.path().join("second-ledger.json");
        ok(applied(&plan_path, &root, &second, &second_ledger));
        assert_eq!(
            fs::read(out.join("sources/a.json")).unwrap(),
            fs::read(second.join("sources/a.json")).unwrap()
        );
        ok(run(&[
            "corpus",
            "verify",
            "--ledger",
            ledger.to_str().unwrap(),
            "--source",
            root.to_str().unwrap(),
            "--bundle",
            out.to_str().unwrap(),
        ]));
        let public = fs::read_to_string(out.join("manifest.json")).unwrap();
        assert!(!public.contains("Alice"));
        assert!(!public.contains(plan["sources"][0]["sha256"].as_str().unwrap()));
        fs::write(out.join("sources/b.json"), "{}").unwrap();
        assert!(
            !run(&[
                "corpus",
                "verify",
                "--ledger",
                ledger.to_str().unwrap(),
                "--source",
                root.to_str().unwrap(),
                "--bundle",
                out.to_str().unwrap()
            ])
            .status
            .success()
        );
    }
}
#[test]
fn source_tampering_incomplete_decisions_and_invalid_scalar_do_not_publish() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("input");
    fs::create_dir(&root).unwrap();
    fs::write(root.join("a.json"), r#"{"id":12,"name":"Alice"}"#).unwrap();
    let path = dir.path().join("plan.json");
    let original = snapshot(&root, &path, "full");
    for variant in [
        "unresolved",
        "tampered",
        "scalar",
        "collision",
        "filename",
        "ledger_inside",
        "version",
    ] {
        let mut plan = original.clone();
        reviewed(&mut plan);
        let source = &mut plan["sources"][0];
        match variant {
            "unresolved" => source["units"][0]["action"] = json!({"kind":"unresolved"}),
            "tampered" => source["units"][1]["original"] = "changed private original".into(),
            "scalar" => {
                source["units"]
                    .as_array_mut()
                    .unwrap()
                    .iter_mut()
                    .find(|u| u["kind"] == "number")
                    .unwrap()["action"] = json!({"kind":"literal","value":"\"text\""})
            }
            "collision" => {
                for u in source["units"]
                    .as_array_mut()
                    .unwrap()
                    .iter_mut()
                    .filter(|u| u["kind"] == "key")
                {
                    u["action"] = json!({"kind":"literal","value":"same"});
                }
            }
            "filename" => {
                source["units"][0]["action"] = json!({"kind":"literal","value":"../escape.json"})
            }
            "version" => plan["version"] = 99.into(),
            _ => {}
        }
        save(&path, &plan);
        let out = dir.path().join(format!("out-{variant}"));
        let ledger = if variant == "ledger_inside" {
            out.join("private.json")
        } else {
            dir.path().join(format!("ledger-{variant}.json"))
        };
        let failure = applied(&path, &root, &out, &ledger);
        assert!(!failure.status.success(), "{variant}");
        assert!(failure.stdout.is_empty());
        assert!(!out.exists());
        assert!(!ledger.exists());
    }
    save(&path, &original);
    fs::write(root.join("a.json"), "{}").unwrap();
    assert!(
        !run(&[
            "corpus",
            "validate",
            path.to_str().unwrap(),
            "--source",
            root.to_str().unwrap()
        ])
        .status
        .success()
    );
}
#[test]
fn duplicate_keys_and_invalid_json_are_not_silently_dropped() {
    for bytes in [
        r#"{"same":1,"same":2}"#,
        r#"{"same":1,"s\u0061me":2}"#,
        r#"{"nested":{"x":1,"x":2}}"#,
        r#"{"name":"\ud800"}"#,
        "\u{a0}\n",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("input");
        fs::create_dir(&root).unwrap();
        fs::write(root.join("a.jsonl"), bytes).unwrap();
        let out = dir.path().join("plan.json");
        let fail = run(&[
            "corpus",
            "inventory",
            root.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
        ]);
        assert!(!fail.status.success(), "{bytes}");
        assert!(!out.exists());
    }
}
#[test]
fn assets_references_manifest_tamper_and_resume_are_accounted() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("input");
    fs::create_dir(&root).unwrap();
    fs::write(
        root.join("a.json"),
        r#"{"asset":"old.bin","type":"image","data":"aW1hZ2U="}"#,
    )
    .unwrap();
    fs::write(root.join("old.bin"), b"owned-original-asset").unwrap();
    let replacement = dir.path().join("replacement.bin");
    let bytes = b"owned-synthetic-replacement";
    fs::write(&replacement, bytes).unwrap();
    use sha2::{Digest, Sha256};
    let hash = format!("{:x}", Sha256::digest(bytes));
    let path = dir.path().join("plan.json");
    let mut plan = snapshot(&root, &path, "full");
    reviewed(&mut plan);
    let asset_id = plan["sources"][1]["id"].clone();
    plan["sources"][1]["units"][0]["action"] = json!({"kind":"literal","value":"new.bin"});
    plan["sources"][1]["units"][1]["action"] =
        json!({"kind":"asset","path":replacement,"sha256":hash});
    for unit in plan["sources"][0]["units"].as_array_mut().unwrap() {
        if unit["original"] == "old.bin" {
            unit["action"] = json!({"kind":"reference","source_id":asset_id});
        }
    }
    save(&path, &plan);
    let out = dir.path().join("out");
    let ledger = dir.path().join("ledger.json");
    ok(applied(&path, &root, &out, &ledger));
    assert_eq!(fs::read(out.join("sources/new.bin")).unwrap(), bytes);
    let doc: Value =
        serde_json::from_slice(&fs::read(out.join("sources/a.json")).unwrap()).unwrap();
    assert_eq!(doc["asset"], "new.bin");
    assert_eq!(doc["data"], "aW1hZ2U=");
    let resumed = ok(run(&[
        "corpus",
        "apply",
        path.to_str().unwrap(),
        "--source",
        root.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
        "--ledger",
        ledger.to_str().unwrap(),
        "--resume",
    ]));
    assert_eq!(resumed["resumed"], true);
    let manifest = out.join("manifest.json");
    let saved = fs::read(&manifest).unwrap();
    fs::write(&manifest, b"{}").unwrap();
    assert!(
        !run(&[
            "corpus",
            "verify",
            "--ledger",
            ledger.to_str().unwrap(),
            "--source",
            root.to_str().unwrap(),
            "--bundle",
            out.to_str().unwrap()
        ])
        .status
        .success()
    );
    fs::write(&manifest, saved).unwrap();
    fs::remove_dir_all(&out).unwrap();
    ok(run(&[
        "corpus",
        "apply",
        path.to_str().unwrap(),
        "--source",
        root.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
        "--ledger",
        ledger.to_str().unwrap(),
        "--resume",
    ]));
    assert_eq!(fs::read(out.join("sources/new.bin")).unwrap(), bytes);
    fs::write(&replacement, b"changed asset").unwrap();
    assert!(
        !run(&[
            "corpus",
            "verify",
            "--ledger",
            ledger.to_str().unwrap(),
            "--source",
            root.to_str().unwrap(),
            "--bundle",
            out.to_str().unwrap()
        ])
        .status
        .success()
    );
}

#[test]
fn iban_surrogates_preserve_country_length_and_checksum() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("input");
    fs::create_dir(&root).unwrap();
    fs::write(
        root.join("a.json"),
        r#"{"account":"DE89 3704 0044 0532 0130 00"}"#,
    )
    .unwrap();
    let path = dir.path().join("plan.json");
    let mut plan = snapshot(&root, &path, "full");
    reviewed(&mut plan);
    plan["sources"][0]["units"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|u| u["kind"] == "string")
        .unwrap()["action"] = json!({"kind":"fake","pattern":"iban"});
    save(&path, &plan);
    let out = dir.path().join("out");
    let ledger = dir.path().join("ledger.json");
    ok(applied(&path, &root, &out, &ledger));
    let doc: Value =
        serde_json::from_slice(&fs::read(out.join("sources/a.json")).unwrap()).unwrap();
    let value = doc["account"].as_str().unwrap();
    assert!(value.starts_with("DE"));
    assert_eq!(value.len(), 27);
    assert_ne!(value, "DE89 3704 0044 0532 0130 00");
    let compact = value.replace(' ', "");
    let reordered = format!("{}{}", &compact[4..], &compact[..4]);
    let mut remainder = 0_u32;
    for byte in reordered.bytes() {
        remainder = if byte.is_ascii_digit() {
            (remainder * 10 + u32::from(byte - b'0')) % 97
        } else {
            (remainder * 100 + u32::from(byte - b'A') + 10) % 97
        };
    }
    assert_eq!(remainder, 1);
}

#[cfg(unix)]
#[test]
fn symlinks_empty_directories_and_source_tree_artifacts_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("input");
    fs::create_dir(&root).unwrap();
    fs::write(root.join("a.txt"), "owned fixture").unwrap();
    let link = root.join("link.txt");
    std::os::unix::fs::symlink(root.join("a.txt"), &link).unwrap();
    let path = dir.path().join("plan.json");
    assert!(
        !run(&[
            "corpus",
            "inventory",
            root.to_str().unwrap(),
            "-o",
            path.to_str().unwrap()
        ])
        .status
        .success()
    );
    assert!(!path.exists());
    fs::remove_file(link).unwrap();
    fs::create_dir(root.join("empty")).unwrap();
    assert!(
        !run(&[
            "corpus",
            "inventory",
            root.to_str().unwrap(),
            "-o",
            path.to_str().unwrap()
        ])
        .status
        .success()
    );
    fs::remove_dir(root.join("empty")).unwrap();
    assert!(
        !run(&[
            "corpus",
            "inventory",
            root.to_str().unwrap(),
            "-o",
            root.join("plan.json").to_str().unwrap()
        ])
        .status
        .success()
    );
}

#[test]
fn span_alias_collapse_is_blocked_and_plain_text_is_not_quoted() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("input");
    fs::create_dir(&root).unwrap();
    fs::write(root.join("a.txt"), "Alice then Bob.").unwrap();
    let path = dir.path().join("plan.json");
    let mut plan = snapshot(&root, &path, "full");
    reviewed(&mut plan);
    plan["sources"][0]["units"][1]["action"] = json!({"kind":"spans","edits":[{"start":0,"end":5,"action":{"kind":"literal","value":"Eve"}},{"start":11,"end":14,"action":{"kind":"literal","value":"Eve"}}]});
    save(&path, &plan);
    let out = dir.path().join("out");
    let ledger = dir.path().join("ledger.json");
    assert!(!applied(&path, &root, &out, &ledger).status.success());
    assert!(!out.exists());
    plan["sources"][0]["units"][1]["action"]["edits"][1]["action"]["value"] = "Sam".into();
    save(&path, &plan);
    ok(applied(&path, &root, &out, &ledger));
    assert_eq!(
        fs::read_to_string(out.join("sources/a.txt")).unwrap(),
        "Eve then Sam."
    );
}

#[test]
fn deny_policy_disqualifies_sources_instead_of_redacting() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("input");
    fs::create_dir(&root).unwrap();
    fs::write(root.join("keep.txt"), "owned synthetic fixture\n").unwrap();
    fs::write(
        root.join("client-x.txt"),
        "confidential: project nightingale\n",
    )
    .unwrap();
    let deny = dir.path().join("deny.txt");
    fs::write(&deny, "# never publish this project\nnightingale\n").unwrap();
    let plan_path = dir.path().join("plan.json");
    let reported = ok(run(&[
        "corpus",
        "inventory",
        root.to_str().unwrap(),
        "-o",
        plan_path.to_str().unwrap(),
        "--deny-file",
        deny.to_str().unwrap(),
    ]));
    assert_eq!(reported["denied_sources"], 1);

    // A denied source can never be approved into a candidate bundle.
    let mut plan: Value = serde_json::from_slice(&fs::read(&plan_path).unwrap()).unwrap();
    reviewed(&mut plan);
    save(&plan_path, &plan);
    let out = dir.path().join("out");
    let ledger = dir.path().join("ledger.json");
    let blocked = applied(&plan_path, &root, &out, &ledger);
    assert!(!blocked.status.success());
    assert!(blocked.stdout.is_empty());
    assert!(!out.exists());
    assert!(!ledger.exists());

    // Removing the disqualified source clears the gate.
    fs::remove_file(root.join("client-x.txt")).unwrap();
    let replan = dir.path().join("plan2.json");
    let clean = ok(run(&[
        "corpus",
        "inventory",
        root.to_str().unwrap(),
        "-o",
        replan.to_str().unwrap(),
        "--deny-file",
        deny.to_str().unwrap(),
    ]));
    assert_eq!(clean["denied_sources"], 0);
    let mut plan: Value = serde_json::from_slice(&fs::read(&replan).unwrap()).unwrap();
    reviewed(&mut plan);
    save(&replan, &plan);
    ok(applied(&replan, &root, &out, &ledger));
}

#[test]
fn invalid_deny_pattern_fails_closed() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("input");
    fs::create_dir(&root).unwrap();
    fs::write(root.join("a.txt"), "owned synthetic fixture\n").unwrap();
    let deny = dir.path().join("deny.txt");
    fs::write(&deny, "unclosed(field\n").unwrap();
    let plan_path = dir.path().join("plan.json");
    let failed = run(&[
        "corpus",
        "inventory",
        root.to_str().unwrap(),
        "-o",
        plan_path.to_str().unwrap(),
        "--deny-file",
        deny.to_str().unwrap(),
    ]);
    assert!(!failed.status.success());
    assert!(!plan_path.exists());
}

#[test]
fn unchanged_sources_require_an_explicit_verbatim_declaration() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("input");
    fs::create_dir(&root).unwrap();
    fs::write(root.join("a.txt"), "owned synthetic fixture\n").unwrap();
    let path = dir.path().join("plan.json");
    let mut plan = snapshot(&root, &path, "session");
    // Approve everything but retain it verbatim, without declaring that choice.
    for source in plan["sources"].as_array_mut().unwrap() {
        source["review"] = json!({"basis":"synthetic","evidence":"fixture","privacy_review":"fixture","serialization_reviewed":true,"media_reviewed":true});
        for unit in source["units"].as_array_mut().unwrap() {
            unit["approval"] = "fixture".into();
            unit["action"] = json!({"kind":"retain"});
        }
    }
    save(&path, &plan);
    let out = dir.path().join("out");
    let ledger = dir.path().join("ledger.json");
    let refused = applied(&path, &root, &out, &ledger);
    assert!(
        !refused.status.success(),
        "unchanged source must not publish silently"
    );
    assert!(!out.exists() && !ledger.exists());
    let message = String::from_utf8_lossy(&refused.stderr);
    assert!(message.contains("publish_verbatim"), "{message}");

    // Declaring it is enough, and the hash stays available for label binding.
    for source in plan["sources"].as_array_mut().unwrap() {
        source["review"]["publish_verbatim"] = true.into();
    }
    save(&path, &plan);
    ok(applied(&path, &root, &out, &ledger));
    let manifest: Value =
        serde_json::from_slice(&fs::read(out.join("manifest.json")).unwrap()).unwrap();
    assert!(manifest["label_sources"][0]["sha256"].is_string());
}

/// Every key nym writes must be declared, because the schema forbids extras.
fn assert_schema_declares(plan: &Value) {
    let schema: Value =
        serde_json::from_str(include_str!("../examples/corpus.schema.json")).unwrap();
    let defs = &schema["$defs"];
    let declared = |def: &str, value: &Value| {
        for key in value.as_object().unwrap().keys() {
            assert!(
                defs[def]["properties"].get(key).is_some(),
                "schema {def} does not declare {key}"
            );
        }
    };
    let kinds: Vec<&str> = defs["action"]["oneOf"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["properties"]["kind"]["const"].as_str().unwrap())
        .collect();
    fn actions<'a>(action: &'a Value, out: &mut Vec<&'a Value>) {
        out.push(action);
        for edit in action["edits"].as_array().into_iter().flatten() {
            actions(&edit["action"], out);
        }
    }
    declared("plan", plan);
    if let Some(proposed) = plan.get("proposed") {
        declared("proposed", proposed);
    }
    for source in plan["sources"].as_array().unwrap() {
        declared("source", source);
        if !source["review"].is_null() {
            declared("review", &source["review"]);
        }
        for unit in source["units"].as_array().unwrap() {
            declared("unit", unit);
            if let Some(proposal) = unit.get("proposal") {
                declared("proposal", proposal);
            }
            let mut found = Vec::new();
            actions(&unit["action"], &mut found);
            for action in found {
                assert!(kinds.contains(&action["kind"].as_str().unwrap()));
            }
        }
    }
}

#[test]
fn reviewed_values_become_approved_placeholders_without_unit_by_unit_review() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("input");
    fs::create_dir_all(root.join("project")).unwrap();
    let email = "alice.example@example.invalid";
    fs::write(
        root.join("project/a.jsonl"),
        format!(
            "{{\"role\":\"user\",\"content\":\"mail {email} and ask Alice Example\"}}\n{{\"role\":\"toolResult\",\"content\":\"to: {email}\",\"n\":3}}\n"
        ),
    )
    .unwrap();
    fs::write(
        root.join("project/b.jsonl"),
        format!("{{\"role\":\"user\",\"content\":\"Bob Sample cc {email}\"}}\n"),
    )
    .unwrap();
    let values = dir.path().join("values.json");
    let session =
        |name: &str, candidates: Value| json!({"session": name, "candidates": candidates});
    let a = session(
        "project/a.jsonl",
        json!([{"value": email, "class": "email"}, {"value": "Alice Example", "class": "person"}]),
    );
    // The same value under another class elsewhere must not get a second token.
    let b = session(
        "project/b.jsonl",
        json!([{"value": "Bob Sample", "class": "person"}, {"value": email, "class": "username"}]),
    );
    save(&values, &json!({"sessions": [a.clone()]}));
    let p = |name: &str| dir.path().join(name);
    let s = |path: &Path| path.to_str().unwrap().to_owned();
    snapshot(&root, &p("plan.json"), "full");

    // Uncovered sources stay unresolved and block, even after bulk approval.
    let summary = ok(run(&[
        "corpus",
        "propose",
        &s(&p("plan.json")),
        "--source",
        &s(&root),
        "--values",
        &s(&values),
        "-o",
        &s(&p("partial.json")),
        "--json",
    ]));
    assert_eq!(
        (
            summary["covered_sources"].as_u64(),
            summary["uncovered_sources"].as_u64()
        ),
        (Some(1), Some(1))
    );
    ok(run(&[
        "corpus",
        "approve",
        &s(&p("partial.json")),
        "-o",
        &s(&p("partial-approved.json")),
        "--receipt",
        "fixture",
        "--all-classes",
        "--unflagged",
        "--renames",
        "--json",
    ]));
    ok(run(&[
        "corpus",
        "review",
        &s(&p("partial-approved.json")),
        "-o",
        &s(&p("partial-reviewed.json")),
        "--basis",
        "synthetic",
        "--evidence",
        "fixture",
        "--privacy-review",
        "fixture",
        "--serialization-reviewed",
        "--json",
    ]));
    let blocked = run(&[
        "corpus",
        "validate",
        &s(&p("partial-reviewed.json")),
        "--source",
        &s(&root),
        "--json",
    ]);
    assert!(!blocked.status.success());
    assert!(String::from_utf8_lossy(&blocked.stdout).contains("unit decisions unresolved"));

    save(&values, &json!({"sessions": [a, b]}));
    ok(run(&[
        "corpus",
        "propose",
        &s(&p("plan.json")),
        "--source",
        &s(&root),
        "--values",
        &s(&values),
        "-o",
        &s(&p("proposed.json")),
        "--rename-files",
        "--json",
    ]));
    let proposed: Value = serde_json::from_slice(&fs::read(p("proposed.json")).unwrap()).unwrap();
    assert_schema_declares(&proposed);
    let units: Vec<&Value> = proposed["sources"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|s| s["units"].as_array().unwrap())
        .collect();
    assert!(
        units.iter().all(|u| u["approval"].is_null()),
        "proposals must not approve"
    );

    // Approving one class leaves the others pending, and typos are reported.
    let partial = ok(run(&[
        "corpus",
        "approve",
        &s(&p("proposed.json")),
        "-o",
        &s(&p("emails.json")),
        "--receipt",
        "fixture",
        "--class",
        "email",
        "--class",
        "emial",
        "--json",
    ]));
    assert_eq!(partial["unknown_classes"], json!(["emial"]));
    assert!(partial["pending_classes"]["person"].as_u64().unwrap() > 0);

    ok(run(&[
        "corpus",
        "approve",
        &s(&p("proposed.json")),
        "-o",
        &s(&p("approved.json")),
        "--receipt",
        "fixture: classes and retention approved",
        "--all-classes",
        "--unflagged",
        "--renames",
        "--json",
    ]));
    ok(run(&[
        "corpus",
        "review",
        &s(&p("approved.json")),
        "-o",
        &s(&p("reviewed.json")),
        "--basis",
        "synthetic",
        "--evidence",
        "fixture",
        "--privacy-review",
        "fixture",
        "--serialization-reviewed",
        "--json",
    ]));
    let reviewed: Value = serde_json::from_slice(&fs::read(p("reviewed.json")).unwrap()).unwrap();
    assert_schema_declares(&reviewed);
    let status = ok(run(&[
        "corpus",
        "status",
        &s(&p("reviewed.json")),
        "--json",
    ]));
    assert_eq!(
        (
            status["unresolved"].as_u64(),
            status["unreviewed_sources"].as_u64()
        ),
        (Some(0), Some(0))
    );

    let out = p("bundle");
    let ledger = p("ledger.json");
    ok(applied(&p("reviewed.json"), &root, &out, &ledger));
    ok(run(&[
        "corpus",
        "verify",
        "--ledger",
        &s(&ledger),
        "--source",
        &s(&root),
        "--bundle",
        &s(&out),
        "--json",
    ]));
    let a_out = fs::read_to_string(out.join("sources/f00000.jsonl")).unwrap();
    let b_out = fs::read_to_string(out.join("sources/f00001.jsonl")).unwrap();
    for text in [&a_out, &b_out] {
        assert!(!text.contains(email) && !text.contains("Alice") && !text.contains("Bob"));
        assert!(
            !text.contains("project"),
            "renamed sources must not keep directory names"
        );
        for line in text.lines() {
            serde_json::from_str::<Value>(line).unwrap();
        }
    }
    // Full scope: one value, one token, in every file and under one class.
    assert_eq!(a_out.matches("<EMAIL_1>").count(), 2);
    assert!(a_out.contains("<PERSON_1>"));
    assert!(b_out.contains("<PERSON_2>") && b_out.contains("<EMAIL_1>"));
    assert!(!b_out.contains("USERNAME"));
}

#[test]
fn reviewed_values_apply_across_sessions_unless_scoped_to_their_session() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("input");
    fs::create_dir(&root).unwrap();
    fs::write(root.join("a.jsonl"), "{\"text\":\"Quillon wrote this\"}\n").unwrap();
    // Detection missed the name here; the review of session a still applies.
    fs::write(root.join("b.jsonl"), "{\"text\":\"ask Quillon later\"}\n").unwrap();
    let values = dir.path().join("values.json");
    save(
        &values,
        &json!({"sessions": [
            {"session": "a.jsonl", "candidates": [{"value": "Quillon", "class": "first_name"}]},
            {"session": "b.jsonl", "candidates": []}
        ]}),
    );
    let p = |name: &str| dir.path().join(name).to_str().unwrap().to_owned();
    snapshot(&root, Path::new(&p("plan.json")), "session");
    let mut texts = Vec::new();
    for scope in ["corpus", "session"] {
        let proposed = p(&format!("proposed-{scope}.json"));
        let summary = ok(run(&[
            "corpus",
            "propose",
            &p("plan.json"),
            "--source",
            root.to_str().unwrap(),
            "--values",
            values.to_str().unwrap(),
            "-o",
            &proposed,
            "--values-scope",
            scope,
            "--json",
        ]));
        assert_eq!(summary["uncovered_sources"], 0);
        let plan: Value = serde_json::from_slice(&fs::read(&proposed).unwrap()).unwrap();
        assert_eq!(plan["proposed"]["values_scope"], scope);
        let b = &plan["sources"][1]["units"];
        texts.push(
            b.as_array()
                .unwrap()
                .iter()
                .any(|u| u["proposal"]["basis"] == "values"),
        );
    }
    assert_eq!(texts, vec![true, false]);
}

#[test]
fn replaced_fields_keep_id_links_and_stripped_media_needs_no_media_review() {
    use base64::Engine;
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("input");
    fs::create_dir(&root).unwrap();
    let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
    let lines = [
        json!({"id":"a1b2c3d4","parentId":null,"message":{"role":"assistant","content":[
            {"type":"thinking","thinking":"plan","thinkingSignature":"EqQBCkYIBxgCKkDsecretsig"},
            {"type":"toolCall","id":"toolu_01XYZ","name":"read"}]}}),
        json!({"id":"e5f6a7b8","parentId":"a1b2c3d4","message":{"role":"toolResult","toolCallId":"toolu_01XYZ","content":[
            {"type":"image","mimeType":"image/png","data":png},
            {"type":"image","mimeType":"image/jpeg","data":png}]}}),
        json!({"type":"leaf","currentLeafId":"e5f6a7b8"}),
    ];
    let text: String = lines.iter().map(|l| format!("{l}\n")).collect();
    fs::write(root.join("s.jsonl"), text).unwrap();
    let values = dir.path().join("values.json");
    save(
        &values,
        &json!({"sessions": [{"session": "s.jsonl", "candidates": []}]}),
    );
    let p = |name: &str| dir.path().join(name).to_str().unwrap().to_owned();
    snapshot(&root, Path::new(&p("plan.json")), "session");
    let summary = ok(run(&[
        "corpus",
        "propose",
        &p("plan.json"),
        "--source",
        root.to_str().unwrap(),
        "--values",
        values.to_str().unwrap(),
        "-o",
        &p("proposed.json"),
        "--replace-field",
        "thinkingSignature,id,parentId,toolCallId",
        "--strip-media",
        "--json",
    ]));
    assert_eq!(summary["proposed_units"]["media_replaced"], 2);
    let proposed: Value = serde_json::from_slice(&fs::read(p("proposed.json")).unwrap()).unwrap();
    assert_schema_declares(&proposed);
    ok(run(&[
        "corpus",
        "approve",
        &p("proposed.json"),
        "-o",
        &p("approved.json"),
        "--receipt",
        "fixture",
        "--unflagged",
        "--fields",
        "--media",
        "--json",
    ]));
    // No --media-reviewed: replaced images keep nothing of the original.
    ok(run(&[
        "corpus",
        "review",
        &p("approved.json"),
        "-o",
        &p("reviewed.json"),
        "--basis",
        "synthetic",
        "--evidence",
        "fixture",
        "--privacy-review",
        "fixture",
        "--serialization-reviewed",
        "--json",
    ]));
    let out = dir.path().join("bundle");
    ok(applied(
        Path::new(&p("reviewed.json")),
        &root,
        &out,
        Path::new(&p("ledger.json")),
    ));
    let text = fs::read_to_string(out.join("sources/s.jsonl")).unwrap();
    for original in ["secretsig", "toolu_01XYZ", "a1b2c3d4", "e5f6a7b8", png] {
        assert!(!text.contains(original));
    }
    let r: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let call = &r[0]["message"]["content"][1]["id"];
    assert_eq!(call, &r[1]["message"]["toolCallId"], "tool call link lost");
    assert_eq!(r[0]["id"], r[1]["parentId"], "parent link lost");
    assert_eq!(
        r[1]["id"], r[2]["currentLeafId"],
        "an equal value elsewhere must share the token"
    );
    assert!(
        r[0]["message"]["content"][0]["thinkingSignature"]
            .as_str()
            .unwrap()
            .starts_with("<THINKING_SIGNATURE_")
    );
    let images = &r[1]["message"]["content"];
    let decode = |i: usize| {
        base64::engine::general_purpose::STANDARD
            .decode(images[i]["data"].as_str().unwrap())
            .unwrap()
    };
    assert!(decode(0).starts_with(b"\x89PNG") && decode(1).starts_with(&[0xff, 0xd8]));
}
