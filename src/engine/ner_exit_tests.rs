//! Subprocess tests exercise libtest and native teardown without downloading models.

use super::{set_exit_code, setup_macos_crash_workaround};

const CHILD: &str = "NYM_TEST_NATIVE_EXIT_CHILD";

// ONNX IR8/opset13: one float input x[1], Identity, one float output y[1].
const IDENTITY_MODEL: &[u8] = &[
    0x08, 0x08, 0x12, 0x08, 0x6e, 0x79, 0x6d, 0x2d, 0x74, 0x65, 0x73, 0x74, 0x3a, 0x4c, 0x0a, 0x10,
    0x0a, 0x01, 0x78, 0x12, 0x01, 0x79, 0x22, 0x08, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79,
    0x12, 0x16, 0x73, 0x79, 0x6e, 0x74, 0x68, 0x65, 0x74, 0x69, 0x63, 0x5f, 0x65, 0x78, 0x69, 0x74,
    0x5f, 0x66, 0x69, 0x78, 0x74, 0x75, 0x72, 0x65, 0x5a, 0x0f, 0x0a, 0x01, 0x78, 0x12, 0x0a, 0x0a,
    0x08, 0x08, 0x01, 0x12, 0x04, 0x0a, 0x02, 0x08, 0x01, 0x62, 0x0f, 0x0a, 0x01, 0x79, 0x12, 0x0a,
    0x0a, 0x08, 0x08, 0x01, 0x12, 0x04, 0x0a, 0x02, 0x08, 0x01, 0x42, 0x04, 0x0a, 0x00, 0x10, 0x0d,
];

fn exercise_native_teardown() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    setup_macos_crash_workaround();
    let directory = tempfile::tempdir()?;
    let model = directory.path().join("identity.onnx");
    std::fs::write(&model, IDENTITY_MODEL)?;
    ort::environment::get_environment()?;
    // Match the real detectors' intentional session leak, not an unsafe Drop.
    let session = std::mem::ManuallyDrop::new(
        ort::session::Session::builder()?
            .with_intra_threads(1)?
            .with_execution_providers([
                ort::execution_providers::CPUExecutionProvider::default().build()
            ])?
            .commit_from_file(model)?,
    );
    let input = ort::value::Tensor::from_array((vec![1_i64], vec![42.0_f32]))?;
    let outputs = session.run(ort::inputs!["x" => input]?)?;
    assert_eq!(outputs["y"].try_extract_raw_tensor::<f32>()?.1, [42.0]);
    Ok(())
}

#[test]
fn harness_exit_status_survives_native_teardown()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if let Ok(mode) = std::env::var(CHILD) {
        exercise_native_teardown()?;
        match mode.as_str() {
            "fail" => panic!("synthetic failing harness"),
            "caught" => {
                assert!(std::panic::catch_unwind(|| panic!("synthetic caught panic")).is_err())
            }
            "0" | "1" | "2" => {
                let code = mode.parse()?;
                set_exit_code(code);
                std::process::exit(code);
            }
            "pass" => {}
            _ => panic!("unexpected native exit fixture mode"),
        }
        return Ok(());
    }
    let expected = [
        ("pass", Some(0)),
        ("fail", Some(101)),
        ("caught", Some(0)),
        ("expected", Some(0)),
        ("0", Some(0)),
        ("1", Some(1)),
        ("2", Some(2)),
    ];
    let mut statuses = Vec::new();
    for (mode, _) in expected {
        let name = if mode == "expected" {
            "engine::ner::exit_tests::expected_panic_keeps_success"
        } else {
            "engine::ner::exit_tests::harness_exit_status_survives_native_teardown"
        };
        let output = std::process::Command::new(std::env::current_exe()?)
            .args(["--exact", name, "--nocapture"])
            .env(CHILD, mode)
            .output()?;
        if mode == "fail" {
            assert!(String::from_utf8_lossy(&output.stdout).contains("test result: FAILED"));
        } else if ["pass", "caught", "expected"].contains(&mode) {
            assert!(String::from_utf8_lossy(&output.stdout).contains("test result: ok"));
        }
        statuses.push((mode, output.status.code()));
    }
    assert_eq!(statuses, expected.to_vec());
    Ok(())
}

#[test]
#[should_panic(expected = "synthetic expected panic")]
fn expected_panic_keeps_success() {
    exercise_native_teardown().expect("synthetic native setup must succeed");
    panic!("synthetic expected panic");
}
