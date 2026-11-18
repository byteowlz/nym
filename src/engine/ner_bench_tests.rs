//! Timed NER throughput harness: measures `TokenClassDetector::detect_batch`
//! over raw JSONL records as a stress workload. This does NOT reproduce traces'
//! decoded text extraction, line selection or deduplication; compare end-to-end
//! speed using the traces CLI scorecard, not this stress test. Ignored by default; run
//! with NYM_TEST_CACHED_NER_MODEL=<model dir> and
//! NYM_BENCH_NER_CORPUS=<dir with *.jsonl>.

use std::time::Instant;

use super::TokenClassDetector;

#[test]
#[ignore = "requires NYM_TEST_CACHED_NER_MODEL and NYM_BENCH_NER_CORPUS; never downloads"]
fn bench_detect_batch_throughput() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = std::env::var("NYM_TEST_CACHED_NER_MODEL")?;
    let corpus = std::env::var("NYM_BENCH_NER_CORPUS")?;
    let provider = match std::env::var("NYM_NER_PROVIDER").as_deref() {
        Ok("cpu") => crate::config::NerProvider::Cpu,
        _ => crate::config::NerProvider::Auto,
    };
    let detector = TokenClassDetector::from_dir(model, Some(0.5), provider, None)?;

    // Deliberately stress the backend with raw records, including metadata.
    let files = walkdir(&corpus);
    let file_count = files.len();
    let mut lines: Vec<String> = Vec::new();
    for entry in files {
        let content = std::fs::read_to_string(&entry)?;
        for line in content.lines() {
            if !line.trim().is_empty() {
                lines.push(line.to_string());
            }
        }
    }
    let total_bytes: usize = lines.iter().map(|l| l.len()).sum();
    eprintln!(
        "records: {} lines, {} bytes in {} files",
        lines.len(),
        total_bytes,
        file_count
    );

    // Warm up (compile buckets on the GPU path), then time.
    let refs: Vec<&str> = lines.iter().take(8).map(String::as_str).collect();
    detector.detect_batch(&refs)?;
    let start = Instant::now();
    let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    let mut total_matches = 0usize;
    for chunk in refs.chunks(64) {
        let found = detector.detect_batch(chunk)?;
        total_matches += found.iter().map(Vec::len).sum::<usize>();
    }
    let elapsed = start.elapsed();
    eprintln!(
        "provider={:?} seconds={:.2} bytes={} MB/s={:.1} matches={}",
        provider,
        elapsed.as_secs_f64(),
        total_bytes,
        total_bytes as f64 / 1e6 / elapsed.as_secs_f64(),
        total_matches
    );
    Ok(())
}

fn walkdir(root: &str) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![std::path::PathBuf::from(root)];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "jsonl") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}
