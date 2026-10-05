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

/// A processed line with its anonymized content and replacements.
#[cfg(feature = "streaming")]
#[derive(Debug, Clone)]
pub struct ProcessedLine {
    /// The anonymized line content
    pub content: String,
    /// Replacements made on this line
    pub replacements: Vec<Replacement>,
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
pub fn process_stream<'a, R>(
    reader: R,
    config: StreamConfig,
) -> impl Stream<Item = StreamResult<ProcessedLine>> + 'a
where
    R: AsyncRead + Unpin + Send + 'a,
{
    stream! {
        let detector = Detector::new(&config.detector_config);
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
                    let matches = detector.detect(&line);

                    if matches.is_empty() {
                        // No PII found, yield line unchanged
                        yield Ok(ProcessedLine {
                            content: line,
                            replacements: vec![],
                        });
                    } else {
                        // Replace PII
                        let mut replacer_guard = replacer.lock().await;
                        let (anonymized, replacements) = replacer_guard.replace_all(&line, &matches);
                        drop(replacer_guard);

                        yield Ok(ProcessedLine {
                            content: anonymized,
                            replacements,
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
    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;

    let mut writer = writer;
    let mut stats = StreamStats::default();

    if config.format == StreamFormat::Json {
        // JSONL: each line is a JSON document. Anonymize its string values
        // and emit one JSON document per line.
        let detector = Detector::new(&config.detector_config);
        let mut replacer = Replacer::new(config.replacer_config.clone());
        if let Some(ref session_id) = config.session_id {
            replacer = replacer.with_session_id(session_id.clone());
        }
        replacer.seed_mappings(&config.seed_mappings);
        let mut lines = BufReader::new(reader).lines();
        while let Some(line) = lines.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }
            stats.lines_processed += 1;
            let (anonymized, replacements, coverage) = crate::engine::process_json_with_selector(
                &line,
                &detector,
                &mut replacer,
                &config.path_selector,
            )
            .map_err(|e| StreamError::Detection(format!("JSON parse failed: {e}")))?;
            if config.json_coverage {
                crate::print_coverage(&coverage);
            }
            stats.pii_found += replacements.len();
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

    let stream = process_stream(reader, config);
    futures::pin_mut!(stream);

    while let Some(result) = stream.next().await {
        match result {
            Ok(processed) => {
                stats.lines_processed += 1;
                stats.pii_found += processed.replacements.len();
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
    /// All replacement mappings collected while streaming (so the caller can
    /// persist them safely through the keyfile module).
    pub replacements: Vec<Replacement>,
}

/// Process a reader to a writer for detection only (no replacement).
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
    use tokio::io::AsyncWriteExt;

    let mut writer = writer;
    let detector = Detector::new(&config.detector_config);
    let buf_reader = BufReader::new(reader);
    let mut lines = buf_reader.lines();

    let mut stats = StreamStats::default();
    let mut line_number = 0usize;

    loop {
        match lines.next_line().await {
            Ok(Some(line)) => {
                line_number += 1;
                stats.lines_processed += 1;

                let matches = detector.detect(&line);

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
