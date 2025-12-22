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
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
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

/// Configuration for streaming processing.
#[cfg(feature = "streaming")]
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct StreamConfig {
    /// Detector configuration
    pub detector_config: DetectorConfig,
    /// Replacer configuration
    pub replacer_config: ReplacerConfig,
    /// Session ID for tracking
    pub session_id: Option<String>,
}

#[cfg(feature = "streaming")]

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

/// Process stdin to stdout with streaming.
///
/// This is the main entry point for streaming mode. It reads from stdin,
/// processes each line, writes to stdout, and optionally writes replacements
/// to a key file.
#[cfg(feature = "streaming")]
pub async fn stream_anon<W>(
    config: StreamConfig,
    mut key_writer: Option<W>,
) -> StreamResult<StreamStats>
where
    W: AsyncWrite + Unpin + Send,
{
    use futures::StreamExt;
    use tokio::io::{stdin, stdout};

    let stdin = stdin();
    let mut stdout = stdout();

    let stream = process_stream(stdin, config);
    futures::pin_mut!(stream);

    let mut stats = StreamStats::default();

    while let Some(result) = stream.next().await {
        match result {
            Ok(processed) => {
                stats.lines_processed += 1;
                stats.pii_found += processed.replacements.len();

                // Write anonymized line to stdout
                stdout.write_all(processed.content.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;

                // Write replacements to key file if provided
                if let Some(ref mut writer) = key_writer {
                    for replacement in &processed.replacements {
                        let json = serde_json::to_string(replacement)
                            .map_err(|e| StreamError::Detection(e.to_string()))?;
                        writer.write_all(json.as_bytes()).await?;
                        writer.write_all(b"\n").await?;
                    }
                }
            }
            Err(e) => {
                return Err(e);
            }
        }
    }

    // Flush key writer
    if let Some(ref mut writer) = key_writer {
        writer.flush().await?;
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
}

/// Process stdin to stdout for detection only (no replacement).
#[cfg(feature = "streaming")]
pub async fn stream_detect(config: StreamConfig) -> StreamResult<StreamStats> {
    use tokio::io::{stdin, stdout};

    let stdin = stdin();
    let mut stdout = stdout();

    let detector = Detector::new(&config.detector_config);
    let buf_reader = BufReader::new(stdin);
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
                        stdout.write_all(output.as_bytes()).await?;
                    }
                    stdout.flush().await?;
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
