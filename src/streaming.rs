//! Streaming PII processing for stdin/stdout pipelines.
//!
//! This module provides async streaming support for processing large files
//! or piped data without loading everything into memory.
//!
//! # Example
//!
//! ```bash
//! # Stream processing with immediate output
//! cat large_file.txt | nym anon --stream > anonymized.txt
//!
//! # Process output from another command
//! some_command | nym anon --stream | another_command
//! ```

#[cfg(feature = "streaming")]
use async_stream::stream;
#[cfg(feature = "streaming")]
use futures::Stream;
#[cfg(feature = "streaming")]
use std::sync::Arc;
#[cfg(feature = "streaming")]
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, BufReader};
#[cfg(feature = "streaming")]
use tokio::sync::Mutex;

#[cfg(feature = "streaming")]
use crate::engine::{Detector, DetectorConfig, Replacement, Replacer, ReplacerConfig};

/// Result type for streaming operations.
#[cfg(feature = "streaming")]
pub type StreamResult<T> = Result<T, StreamError>;

/// Errors that can occur during streaming.
#[cfg(feature = "streaming")]
#[derive(Debug)]
pub enum StreamError {
    /// I/O error during read/write
    Io(std::io::Error),
    /// Detection error
    Detection(String),
}

#[cfg(feature = "streaming")]
impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamError::Io(e) => write!(f, "I/O error: {e}"),
            StreamError::Detection(e) => write!(f, "Detection error: {e}"),
        }
    }
}

#[cfg(feature = "streaming")]
impl std::error::Error for StreamError {}

#[cfg(feature = "streaming")]
impl From<std::io::Error> for StreamError {
    fn from(e: std::io::Error) -> Self {
        StreamError::Io(e)
    }
}

#[cfg(feature = "streaming")]
impl From<crate::engine::detector::DetectionError> for StreamError {
    fn from(error: crate::engine::detector::DetectionError) -> Self {
        Self::Detection(error.to_string())
    }
}

#[cfg(feature = "streaming")]
fn record_error(line: usize, error: impl std::fmt::Display) -> StreamError {
    // Only privacy-safe engine/format errors may enter this boundary.
    StreamError::Detection(format!("line {line}: {error}"))
}

#[cfg(feature = "streaming")]
fn json_record(line: &str, number: usize) -> StreamResult<&str> {
    if let Some(without_bom) = line.trim_start().strip_prefix('\u{feff}') {
        if number != 1 {
            return Err(record_error(number, "unexpected JSON BOM"));
        }
        return Ok(without_bom);
    }
    Ok(line)
}

/// A processed line with its anonymized content and replacements.
#[cfg(feature = "streaming")]
#[derive(Debug, Clone)]
pub struct ProcessedLine {
    /// The anonymized line content
    pub content: String,
    /// Replacements made on this line
    pub replacements: Vec<Replacement>,
    pub trace_stats: crate::engine::TracePolicyStats,
}

/// Streaming output format.
#[cfg(feature = "streaming")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StreamFormat {
    /// Plain text line-by-line.
    #[default]
    Text,
    /// JSON per record (JSONL): each line is a JSON document whose string
    /// values are anonymized; output is one JSON document per line.
    Json,
}

/// Configuration for streaming processing.
#[cfg(feature = "streaming")]
#[derive(Debug, Clone, Default)]
pub struct StreamConfig {
    /// Detector configuration
    pub detector_config: DetectorConfig,
    /// Replacer configuration
    pub replacer_config: ReplacerConfig,
    /// Session ID for tracking
    pub session_id: Option<String>,
    /// Output format for the stream (default: text).
    pub format: StreamFormat,
    /// Existing replacement mappings to seed the replacer with so consistent/
    /// fake strategies reuse recorded aliases across runs.
    #[cfg(feature = "streaming")]
    pub seed_mappings: Vec<Replacement>,
    /// JSON path selector for JSONL streaming (default: scan everything).
    #[cfg(feature = "streaming")]
    pub path_selector: crate::engine::PathSelector,
    /// Report scanned/skipped paths to stderr for each JSONL document.
    pub json_coverage: bool,
}

/// Creates an async stream that processes lines from a reader.
///
/// Reads lines from the input, detects and replaces PII, and yields
/// processed lines with their replacements.
///
/// # Arguments
///
/// * `reader` - An async reader (e.g., `tokio::io::stdin()`)
/// * `config` - Streaming configuration
///
/// # Returns
///
/// A stream of `ProcessedLine` results.
#[cfg(feature = "streaming")]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "Public line-stream API retained for consumers")
)]
pub fn process_stream<'a, R>(
    reader: R,
    config: StreamConfig,
) -> impl Stream<Item = StreamResult<ProcessedLine>> + 'a
where
    R: AsyncRead + Unpin + Send + 'a,
{
    let detector = Detector::new(&config.detector_config);
    process_stream_with_detector(reader, config, detector)
}

#[cfg(feature = "streaming")]
fn process_stream_with_detector<'a, R>(
    reader: R,
    config: StreamConfig,
    detector: Result<Detector, crate::engine::detector::DetectionError>,
) -> impl Stream<Item = StreamResult<ProcessedLine>> + 'a
where
    R: AsyncRead + Unpin + Send + 'a,
{
    stream! {
        let detector = match detector {
            Ok(detector) => detector,
            Err(error) => { yield Err(error.into()); return; }
        };
        let replacer_config = config.replacer_config.clone();

        let mut replacer = Replacer::new(replacer_config);
        if let Some(ref session_id) = config.session_id {
            replacer = replacer.with_session_id(session_id.clone());
        }
        replacer.seed_mappings(&config.seed_mappings);

        // Wrap replacer in Arc<Mutex> for shared mutable access
        let replacer = Arc::new(Mutex::new(replacer));

        let buf_reader = BufReader::new(reader);
        let mut lines = buf_reader.lines();

        loop {
            match lines.next_line().await {
                Ok(Some(line)) => {
                    // Detect PII in this line
                    let detected = match detector.detect_with_stats(&line) {
                        Ok(detected) => detected,
                        Err(error) => { yield Err(error.into()); return; }
                    };

                    let matches = detected.matches;
                    let trace_stats = detected.stats;
                    if matches.is_empty() {
                        // No PII found, yield line unchanged
                        yield Ok(ProcessedLine {
                            content: line,
                            replacements: vec![],
                            trace_stats,
                        });
                    } else {
                        // Replace PII
                        let mut replacer_guard = replacer.lock().await;
                        let (anonymized, replacements) = replacer_guard.replace_all(&line, &matches);
                        drop(replacer_guard);

                        yield Ok(ProcessedLine {
                            content: anonymized,
                            replacements,
                            trace_stats,
                        });
                    }
                }
                Ok(None) => {
                    // EOF
                    break;
                }
                Err(e) => {
                    yield Err(StreamError::Io(e));
                    break;
                }
            }
        }
    }
}

/// Process a reader to a writer with streaming, collecting replacements.
///
/// Unlike the old stdin/stdout-bound implementation, this accepts explicit
/// reader/writer so file-to-file, file-to-stdout, stdin-to-file and
/// stdin-to-stdout all work. Replacements are collected (not written
/// incrementally) so the caller can persist a key file safely via the keyfile
/// module, which never truncates an existing mapping.
///
/// Earlier completed records may already be written when a later record fails.
/// The failed record is never emitted. Callers requiring atomic file output
/// must stage it and publish only on `Ok`; stdout cannot be rolled back.
#[cfg(feature = "streaming")]
pub async fn stream_anon<R, W>(
    config: StreamConfig,
    reader: R,
    writer: W,
) -> StreamResult<StreamStats>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    let detector = Detector::new(&config.detector_config)?;
    stream_anon_with_detector(config, reader, writer, detector).await
}

#[cfg(feature = "streaming")]
async fn stream_anon_with_detector<R, W>(
    config: StreamConfig,
    reader: R,
    writer: W,
    detector: Detector,
) -> StreamResult<StreamStats>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;
    let mut writer = writer;
    let mut stats = StreamStats::default();

    if config.format == StreamFormat::Json {
        // JSONL: each line is a JSON document. Anonymize its string values
        // and emit one JSON document per line.
        let mut replacer = Replacer::new(config.replacer_config.clone());
        if let Some(ref session_id) = config.session_id {
            replacer = replacer.with_session_id(session_id.clone());
        }
        replacer.seed_mappings(&config.seed_mappings);
        let mut lines = BufReader::new(reader).lines();
        let mut line_number = 0;
        while let Some(line) = lines.next_line().await? {
            line_number += 1;
            let line = json_record(&line, line_number)?;
            if line.trim().is_empty() {
                continue;
            }
            stats.lines_processed += 1;
            let (anonymized, replacements, coverage, trace_stats) =
                crate::engine::formats::process_json_with_stats(
                    &line,
                    &detector,
                    &mut replacer,
                    &config.path_selector,
                )
                .map_err(|error| record_error(line_number, error))?;
            if config.json_coverage {
                crate::print_coverage(&coverage);
            }
            stats.pii_found += replacements.len();
            crate::engine::audit::merge_trace_stats(&mut stats.trace_stats, trace_stats);
            stats.replacements.extend(replacements);
            // JSONL requires exactly one JSON document per line, so compact the
            // pretty-printed output from `process_json`.
            let compact = serde_json::from_str::<serde_json::Value>(&anonymized)
                .map(|v| serde_json::to_string(&v).unwrap_or_else(|_| anonymized.clone()))
                .unwrap_or(anonymized);
            writer.write_all(compact.as_bytes()).await?;
            writer.write_all(b"\n").await?;
            writer.flush().await?;
        }
        return Ok(stats);
    }

    let stream = process_stream_with_detector(reader, config, Ok(detector));
    futures::pin_mut!(stream);

    while let Some(result) = stream.next().await {
        match result {
            Ok(processed) => {
                stats.lines_processed += 1;
                stats.pii_found += processed.replacements.len();
                crate::engine::audit::merge_trace_stats(
                    &mut stats.trace_stats,
                    processed.trace_stats,
                );
                stats.replacements.extend(processed.replacements);

                writer.write_all(processed.content.as_bytes()).await?;
                writer.write_all(b"\n").await?;
                writer.flush().await?;
            }
            Err(e) => {
                return Err(e);
            }
        }
    }

    Ok(stats)
}

/// Statistics from streaming processing.
#[cfg(feature = "streaming")]
#[derive(Debug, Default, Clone)]
pub struct StreamStats {
    /// Number of lines processed
    pub lines_processed: usize,
    /// Total PII occurrences found
    pub pii_found: usize,
    /// Counts only, never values or field names.
    pub trace_stats: crate::engine::TracePolicyStats,
    /// All replacement mappings collected while streaming (so the caller can
    /// persist them safely through the keyfile module).
    pub replacements: Vec<Replacement>,
}

/// Process a reader to a writer for detection only (no replacement).
/// JSON streams scan selected string values, never keys or raw JSON syntax.
/// A failure terminates the stream; earlier completed records may be written,
/// but the failed record contributes no output or successful final statistics.
#[cfg(feature = "streaming")]
pub async fn stream_detect<R, W>(
    config: StreamConfig,
    reader: R,
    writer: W,
) -> StreamResult<StreamStats>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    let detector = Detector::new(&config.detector_config)?;
    stream_detect_with_detector(config, reader, writer, detector).await
}

#[cfg(feature = "streaming")]
async fn stream_detect_with_detector<R, W>(
    config: StreamConfig,
    reader: R,
    writer: W,
    detector: Detector,
) -> StreamResult<StreamStats>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    use tokio::io::AsyncWriteExt;
    let mut writer = writer;
    let buf_reader = BufReader::new(reader);
    let mut lines = buf_reader.lines();

    let mut stats = StreamStats::default();
    let mut line_number = 0usize;

    loop {
        match lines.next_line().await {
            Ok(Some(line)) => {
                line_number += 1;
                let line = if config.format == StreamFormat::Json {
                    json_record(&line, line_number)?
                } else {
                    &line
                };
                if config.format == StreamFormat::Json && line.trim().is_empty() {
                    continue;
                }
                stats.lines_processed += 1;
                let matches = if config.format == StreamFormat::Json {
                    let (matches, coverage, trace_stats) =
                        crate::engine::formats::detect_json_with_stats(
                            line,
                            &detector,
                            &config.path_selector,
                        )
                        .map_err(|error| record_error(line_number, error))?;
                    if config.json_coverage {
                        crate::print_coverage(&coverage);
                    }
                    crate::engine::audit::merge_trace_stats(&mut stats.trace_stats, trace_stats);
                    matches
                        .into_iter()
                        .map(|matched| matched.pii_match)
                        .collect()
                } else {
                    let detected = detector
                        .detect_with_stats(line)
                        .map_err(|error| record_error(line_number, error))?;
                    crate::engine::audit::merge_trace_stats(&mut stats.trace_stats, detected.stats);
                    detected.matches
                };

                if !matches.is_empty() {
                    stats.pii_found += matches.len();

                    // Output matches in a compact format
                    for m in &matches {
                        let output = format!(
                            "{}:{}:{}-{}:[{}] {}\n",
                            line_number,
                            m.pattern_name,
                            m.start,
                            m.end,
                            format!("{:?}", m.confidence).to_lowercase(),
                            m.matched_text
                        );
                        writer.write_all(output.as_bytes()).await?;
                    }
                    writer.flush().await?;
                }
            }
            Ok(None) => break,
            Err(e) => return Err(StreamError::Io(e)),
        }
    }

    Ok(stats)
}

#[cfg(all(test, feature = "streaming"))]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[cfg(feature = "ner")]
    #[tokio::test]
    async fn initialization_failure_writes_nothing_for_any_backend_or_stream_format() {
        use crate::engine::detector::NerBackend;
        let dir = tempfile::tempdir().unwrap();
        for backend in [NerBackend::Gliner, NerBackend::TokenClass, NerBackend::Both] {
            for format in [StreamFormat::Text, StreamFormat::Json] {
                let config = StreamConfig {
                    format,
                    detector_config: DetectorConfig::default()
                        .with_ner(true)
                        .with_ner_backend(backend)
                        .with_ner_model(dir.path().to_string_lossy())
                        .with_ner_token_model(dir.path()),
                    ..Default::default()
                };
                let input = if format == StreamFormat::Text {
                    "Alice Smith\n"
                } else {
                    "{\"name\":\"Alice Smith\"}\n"
                };
                let original = b"existing destination".to_vec();
                let mut output = original.clone();
                assert!(
                    stream_anon(config.clone(), input.as_bytes(), &mut output)
                        .await
                        .is_err()
                );
                assert_eq!(output, original);
                assert!(
                    stream_detect(config.clone(), input.as_bytes(), &mut output)
                        .await
                        .is_err()
                );
                assert_eq!(output, original);
                assert!(
                    crate::streaming_ner::stream_anon_ner(config, input.as_bytes(), &mut output)
                        .await
                        .is_err()
                );
                assert_eq!(output, original);
            }
        }
    }

    #[cfg(feature = "ner")]
    #[tokio::test]
    async fn runtime_failure_never_emits_failed_text_or_json_record() {
        use crate::engine::detector::{NerBackend, tests::injected};
        for backend in [NerBackend::Gliner, NerBackend::TokenClass, NerBackend::Both] {
            for format in [StreamFormat::Text, StreamFormat::Json] {
                let input = if format == StreamFormat::Text {
                    "Completed record\nAlice Smith\n"
                } else {
                    "{\"value\":\"Completed record\"}\n{\"value\":\"Alice Smith\"}\n"
                };
                let config = StreamConfig {
                    format,
                    ..Default::default()
                };
                let mut output = Vec::new();
                let result = stream_anon_with_detector(
                    config.clone(),
                    input.as_bytes(),
                    &mut output,
                    injected(backend, Some("Alice"), false),
                )
                .await;
                assert!(result.is_err());
                assert!(
                    !output.is_empty(),
                    "completed records may already be emitted"
                );
                assert!(!String::from_utf8_lossy(&output).contains("Alice"));
                output.clear();
                let error = stream_detect_with_detector(
                    config,
                    input.as_bytes(),
                    &mut output,
                    injected(backend, Some("Alice"), false),
                )
                .await
                .unwrap_err();
                assert!(!String::from_utf8_lossy(&output).contains("Alice"));
                assert!(!error.to_string().contains("Alice"));
                assert!(!format!("{error:?}").contains("secret-provider-key"));
            }
        }
    }

    #[tokio::test]
    async fn json_stream_respects_structure_selection_bom_and_crlf() {
        let input = "\u{feff}\r\n{\"key@example.invalid\":\"safe\",\"secret\":\"alice@example.invalid\",\"skip\":\"bob@example.invalid\"}\r\n\r\n";
        let config = StreamConfig {
            format: StreamFormat::Json,
            path_selector: crate::engine::PathSelector::new(&[], &["skip".into()]).unwrap(),
            ..Default::default()
        };
        let mut output = Vec::new();
        let stats = stream_detect(config.clone(), input.as_bytes(), &mut output)
            .await
            .unwrap();
        assert_eq!(stats.pii_found, 1);
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("alice@example.invalid"));
        assert!(!text.contains("key@example.invalid"));
        assert!(!text.contains("bob@example.invalid"));
        let mut output = Vec::new();
        let stats = stream_anon(config, input.as_bytes(), &mut output)
            .await
            .unwrap();
        assert_eq!(stats.lines_processed, 1);
        let document: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(document["skip"], "bob@example.invalid");
        assert_eq!(document["key@example.invalid"], "safe");
        assert_ne!(document["secret"], "alice@example.invalid");
    }

    #[tokio::test]
    async fn json_stream_rejects_bom_after_first_physical_line() {
        let input = "{}\n\u{feff}{}\n";
        let config = StreamConfig {
            format: StreamFormat::Json,
            ..Default::default()
        };
        let mut output = Vec::new();
        assert!(
            stream_anon(config.clone(), input.as_bytes(), &mut output)
                .await
                .is_err()
        );
        output.clear();
        assert!(
            stream_detect(config, input.as_bytes(), &mut output)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn malformed_json_stream_diagnostics_are_value_free() {
        let input = "{\"ok\":\"safe\"}\n{\"value\": confidentialCredential}\n";
        let config = StreamConfig {
            format: StreamFormat::Json,
            ..Default::default()
        };
        for detect in [false, true] {
            let mut output = Vec::new();
            let error = if detect {
                stream_detect(config.clone(), input.as_bytes(), &mut output).await
            } else {
                stream_anon(config.clone(), input.as_bytes(), &mut output).await
            }
            .unwrap_err();
            assert!(error.to_string().contains("line 2"));
            assert!(!error.to_string().contains("confidentialCredential"));
            assert!(!format!("{error:?}").contains("confidentialCredential"));
        }
    }

    #[tokio::test]
    async fn trace_counts_cover_text_and_structured_anon_and_detection() {
        for (format, input) in [
            (StreamFormat::Text, "PrivateTerm\n雪 PrivateTerm\n"),
            (
                StreamFormat::Json,
                "{\"PrivateTerm\":\"ordinary\",\"value\":\"PrivateTerm\"}\n{\"value\":\"雪 PrivateTerm\"}\n",
            ),
        ] {
            let config = StreamConfig {
                format,
                detector_config: DetectorConfig::default().with_trace_policy(
                    crate::engine::TracePolicyConfig {
                        sensitive_terms: vec!["PrivateTerm".into()],
                        ..Default::default()
                    },
                ),
                ..Default::default()
            };
            let mut output = Vec::new();
            let stats = stream_detect(config.clone(), input.as_bytes(), &mut output)
                .await
                .unwrap();
            let expected = crate::engine::TracePolicyStats {
                sensitive_term_matches: 2,
                ..Default::default()
            };
            assert_eq!(stats.trace_stats, expected);
            output.clear();
            let stats = stream_anon(config, input.as_bytes(), &mut output)
                .await
                .unwrap();
            assert_eq!(stats.trace_stats, expected);
            assert_eq!(stats.pii_found, 2);
        }
    }

    #[tokio::test]
    async fn test_process_stream_no_pii() {
        let input = b"Hello world\nThis is a test\n";
        let reader = &input[..];

        let config = StreamConfig::default();
        let stream = process_stream(reader, config);
        futures::pin_mut!(stream);

        let mut results = Vec::new();
        while let Some(result) = stream.next().await {
            results.push(result.unwrap());
        }

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].content, "Hello world");
        assert_eq!(results[0].replacements.len(), 0);
        assert_eq!(results[1].content, "This is a test");
    }

    #[tokio::test]
    async fn test_process_stream_with_pii() {
        let input = b"Contact test@example.com for info\nNo PII here\n";
        let reader = &input[..];

        let config = StreamConfig::default();
        let stream = process_stream(reader, config);
        futures::pin_mut!(stream);

        let mut results = Vec::new();
        while let Some(result) = stream.next().await {
            results.push(result.unwrap());
        }

        assert_eq!(results.len(), 2);
        // First line should have replacement
        assert!(!results[0].replacements.is_empty());
        assert!(results[0].content.contains("<EMAIL>") || results[0].content.contains("@"));
        // Second line should be unchanged
        assert_eq!(results[1].content, "No PII here");
        assert!(results[1].replacements.is_empty());
    }
}
