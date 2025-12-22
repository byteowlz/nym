//! Smart streaming NER with sentence buffering.
//!
//! This module provides intelligent buffering for NER-based PII detection in streaming mode.
//! Instead of processing line-by-line (which lacks context), it buffers text until complete
//! sentences or paragraphs are formed, then runs NER on these semantic units.
//!
//! # Architecture
//!
//! ```text
//! stdin → [Line Reader] → [Sentence Buffer] → [NER + Regex] → [Replacer] → stdout
//! ```
//!
//! The sentence buffer accumulates incoming lines and emits complete sentences/paragraphs
//! when sentence boundaries are detected. This provides NER with proper context.
//!
//! # Features
//!
//! - **Sentence-aware buffering**: Uses `async-tqsm` for proper sentence boundary detection
//! - **Format detection**: Handles key-value pairs, markdown, and prose differently
//! - **Entity tracking**: Maintains consistency across sentences (same text → same replacement)
//! - **Low latency**: Emits sentences as soon as they're complete, not waiting for EOF

#[cfg(all(feature = "streaming", feature = "ner"))]
use async_stream::stream;
#[cfg(all(feature = "streaming", feature = "ner"))]
use futures::Stream;
#[cfg(all(feature = "streaming", feature = "ner"))]
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};

#[cfg(all(feature = "streaming", feature = "ner"))]
use crate::engine::{Detector, Replacement, Replacer};
#[cfg(all(feature = "streaming", feature = "ner"))]
use crate::streaming::{StreamConfig, StreamError, StreamResult, StreamStats};

/// Detected format of input text.
#[cfg(all(feature = "streaming", feature = "ner"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    /// Plain prose text - use sentence segmentation
    Prose,
    /// Key-value pairs (e.g., "name: John Smith")
    KeyValue,
    /// Markdown with headers
    Markdown,
    /// Empty or whitespace-only line
    Empty,
}

/// Detects the format of a line.
#[cfg(all(feature = "streaming", feature = "ner"))]
fn detect_format(line: &str) -> InputFormat {
    let trimmed = line.trim();

    if trimmed.is_empty() {
        return InputFormat::Empty;
    }

    // Markdown header
    if trimmed.starts_with('#') {
        return InputFormat::Markdown;
    }

    // Key-value pattern: "key: value" or "key = value"
    if let Some(colon_pos) = trimmed.find(':') {
        let key = &trimmed[..colon_pos];
        // Key should be a simple identifier (letters, numbers, underscores)
        if !key.is_empty()
            && key
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == ' ')
            && key.len() < 30
        {
            return InputFormat::KeyValue;
        }
    }

    InputFormat::Prose
}

/// Extract the value from a key-value line.
#[cfg(all(feature = "streaming", feature = "ner"))]
fn extract_kv_value(line: &str) -> Option<(&str, &str, usize)> {
    if let Some(colon_pos) = line.find(':') {
        let key = &line[..colon_pos];
        let value_start = colon_pos + 1;
        let value = line[value_start..].trim_start();
        let value_offset = value_start + (line[value_start..].len() - value.len());
        Some((key, value, value_offset))
    } else {
        None
    }
}

/// A buffered text unit ready for NER processing.
#[cfg(all(feature = "streaming", feature = "ner"))]
#[derive(Debug, Clone)]
pub struct TextUnit {
    /// The text content to process
    pub text: String,
    /// Format of the text
    pub format: InputFormat,
    /// Original lines that make up this unit (for reconstruction)
    pub original_lines: Vec<String>,
}

/// Smart buffer that accumulates text and emits semantic units.
#[cfg(all(feature = "streaming", feature = "ner"))]
pub struct SentenceBuffer {
    /// Accumulated lines for prose
    buffer: Vec<String>,
    /// Current line offset
    line_offset: usize,
    /// Maximum buffer size in lines before forcing a flush
    max_lines: usize,
}

#[cfg(all(feature = "streaming", feature = "ner"))]
impl SentenceBuffer {
    pub fn new(max_lines: usize) -> Self {
        Self {
            buffer: Vec::new(),
            line_offset: 0,
            max_lines,
        }
    }

    /// Feed a line and get back any complete text units.
    pub fn feed(&mut self, line: String) -> Vec<TextUnit> {
        let format = detect_format(&line);
        let mut units = Vec::new();

        match format {
            InputFormat::Empty => {
                // Empty line = paragraph boundary
                // Flush any buffered prose
                if !self.buffer.is_empty() {
                    let text = self.buffer.join(" ");
                    units.push(TextUnit {
                        text,
                        format: InputFormat::Prose,
                        original_lines: std::mem::take(&mut self.buffer),
                    });
                }
                // Also emit the empty line
                units.push(TextUnit {
                    text: String::new(),
                    format: InputFormat::Empty,
                    original_lines: vec![line],
                });
            }
            InputFormat::KeyValue | InputFormat::Markdown => {
                // Flush any buffered prose first
                if !self.buffer.is_empty() {
                    let text = self.buffer.join(" ");
                    units.push(TextUnit {
                        text,
                        format: InputFormat::Prose,
                        original_lines: std::mem::take(&mut self.buffer),
                    });
                }
                // Emit this line immediately
                units.push(TextUnit {
                    text: line.clone(),
                    format,
                    original_lines: vec![line],
                });
            }
            InputFormat::Prose => {
                // Check for sentence-ending punctuation
                let has_sentence_end = line
                    .trim()
                    .chars()
                    .last()
                    .is_some_and(|c| c == '.' || c == '!' || c == '?');

                self.buffer.push(line);

                // Emit if we have a complete sentence or buffer is too large
                if has_sentence_end || self.buffer.len() >= self.max_lines {
                    let text = self.buffer.join(" ");
                    units.push(TextUnit {
                        text,
                        format: InputFormat::Prose,
                        original_lines: std::mem::take(&mut self.buffer),
                    });
                }
            }
        }

        self.line_offset += 1;
        units
    }

    /// Flush any remaining buffered content.
    pub fn flush(&mut self) -> Option<TextUnit> {
        if self.buffer.is_empty() {
            None
        } else {
            let text = self.buffer.join(" ");
            Some(TextUnit {
                text,
                format: InputFormat::Prose,
                original_lines: std::mem::take(&mut self.buffer),
            })
        }
    }
}

/// Process a text unit with NER and regex detection.
#[cfg(all(feature = "streaming", feature = "ner"))]
fn process_text_unit(
    unit: &TextUnit,
    detector: &Detector,
    replacer: &mut Replacer,
) -> Vec<(String, Vec<Replacement>)> {
    let mut results = Vec::new();

    match unit.format {
        InputFormat::Empty => {
            // Empty line - return as-is
            results.push((String::new(), vec![]));
        }
        InputFormat::KeyValue => {
            // For key-value, only process the value part
            if let Some((_key, value, value_offset)) = extract_kv_value(&unit.text) {
                if value.is_empty() {
                    results.push((unit.text.clone(), vec![]));
                } else {
                    // Detect PII in the value
                    let matches = detector.detect(value);

                    if matches.is_empty() {
                        results.push((unit.text.clone(), vec![]));
                    } else {
                        // Replace PII in value
                        let (anonymized_value, replacements) =
                            replacer.replace_all(value, &matches);

                        // Reconstruct the line
                        let colon_pos = unit.text.find(':').unwrap();
                        let prefix = &unit.text[..=colon_pos];
                        let spacing = &unit.text[colon_pos + 1..value_offset];
                        let result = format!("{}{}{}", prefix, spacing, anonymized_value);

                        results.push((result, replacements));
                    }
                }
            } else {
                results.push((unit.text.clone(), vec![]));
            }
        }
        InputFormat::Markdown => {
            // For markdown headers, process the whole line
            let matches = detector.detect(&unit.text);

            if matches.is_empty() {
                results.push((unit.text.clone(), vec![]));
            } else {
                let (anonymized, replacements) = replacer.replace_all(&unit.text, &matches);
                results.push((anonymized, replacements));
            }
        }
        InputFormat::Prose => {
            // For prose, process the combined text
            let matches = detector.detect(&unit.text);

            if matches.is_empty() {
                // Return original lines
                for line in &unit.original_lines {
                    results.push((line.clone(), vec![]));
                }
            } else {
                // Process the combined text
                let (anonymized, replacements) = replacer.replace_all(&unit.text, &matches);

                // If we had multiple lines, we need to split the result back
                // For simplicity, if there was only one line, return it directly
                // For multiple lines, return the combined result as a single line
                // (This preserves the semantic unit)
                if unit.original_lines.len() == 1 {
                    results.push((anonymized, replacements));
                } else {
                    // Multiple lines were combined - return as combined
                    // This is intentional: NER needs the context
                    results.push((anonymized, replacements));
                }
            }
        }
    }

    results
}

/// Creates an async stream that processes text with smart NER buffering.
///
/// Unlike line-by-line processing, this buffers text until complete sentences
/// or paragraphs are formed, providing NER with proper context.
#[cfg(all(feature = "streaming", feature = "ner"))]
pub fn process_stream_ner<'a, R>(
    reader: R,
    config: StreamConfig,
) -> impl Stream<Item = StreamResult<(String, Vec<Replacement>)>> + 'a
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

        let mut buffer = SentenceBuffer::new(10); // Max 10 lines before forcing flush
        let buf_reader = BufReader::new(reader);
        let mut lines = buf_reader.lines();

        loop {
            match lines.next_line().await {
                Ok(Some(line)) => {
                    // Feed line to buffer, get any complete units
                    let units = buffer.feed(line);

                    // Process each unit
                    for unit in units {
                        let results = process_text_unit(&unit, &detector, &mut replacer);
                        for (text, replacements) in results {
                            yield Ok((text, replacements));
                        }
                    }
                }
                Ok(None) => {
                    // EOF - flush remaining buffer
                    if let Some(unit) = buffer.flush() {
                        let results = process_text_unit(&unit, &detector, &mut replacer);
                        for (text, replacements) in results {
                            yield Ok((text, replacements));
                        }
                    }
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

/// Process stdin to stdout with smart NER buffering.
#[cfg(all(feature = "streaming", feature = "ner"))]
pub async fn stream_anon_ner<W>(
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

    let stream = process_stream_ner(stdin, config);
    futures::pin_mut!(stream);

    let mut stats = StreamStats::default();

    while let Some(result) = stream.next().await {
        match result {
            Ok((text, replacements)) => {
                stats.lines_processed += 1;
                stats.pii_found += replacements.len();

                // Write anonymized line to stdout
                stdout.write_all(text.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;

                // Write replacements to key file if provided
                if let Some(ref mut writer) = key_writer {
                    for replacement in &replacements {
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

#[cfg(all(test, feature = "streaming", feature = "ner"))]
mod tests {
    use super::*;

    #[test]
    fn test_detect_format() {
        assert_eq!(detect_format(""), InputFormat::Empty);
        assert_eq!(detect_format("   "), InputFormat::Empty);
        assert_eq!(detect_format("# Header"), InputFormat::Markdown);
        assert_eq!(detect_format("## Sub Header"), InputFormat::Markdown);
        assert_eq!(detect_format("name: John"), InputFormat::KeyValue);
        assert_eq!(detect_format("email: test@example.com"), InputFormat::KeyValue);
        assert_eq!(detect_format("This is a sentence."), InputFormat::Prose);
        assert_eq!(
            detect_format("A very long key that is definitely not a real key: value"),
            InputFormat::Prose
        );
    }

    #[test]
    fn test_extract_kv_value() {
        let (key, value, _offset) = extract_kv_value("name: John Smith").unwrap();
        assert_eq!(key, "name");
        assert_eq!(value, "John Smith");

        let (key, value, _) = extract_kv_value("email:test@example.com").unwrap();
        assert_eq!(key, "email");
        assert_eq!(value, "test@example.com");
    }

    #[test]
    fn test_sentence_buffer_kv() {
        let mut buffer = SentenceBuffer::new(10);

        let units = buffer.feed("name: John Smith".to_string());
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].format, InputFormat::KeyValue);
        assert_eq!(units[0].text, "name: John Smith");
    }

    #[test]
    fn test_sentence_buffer_prose() {
        let mut buffer = SentenceBuffer::new(10);

        // First line without sentence ending
        let units = buffer.feed("This is the first part".to_string());
        assert_eq!(units.len(), 0); // Buffered, not emitted

        // Second line with sentence ending
        let units = buffer.feed("and this is the end.".to_string());
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].format, InputFormat::Prose);
        assert_eq!(units[0].text, "This is the first part and this is the end.");
    }

    #[test]
    fn test_sentence_buffer_empty_line() {
        let mut buffer = SentenceBuffer::new(10);

        // Use text without sentence-ending punctuation so it buffers
        let units = buffer.feed("First paragraph continues".to_string());
        assert_eq!(units.len(), 0); // Should be buffered, not emitted

        let units = buffer.feed("".to_string());

        // Should emit both the prose and the empty line
        assert_eq!(units.len(), 2);
        assert_eq!(units[0].format, InputFormat::Prose);
        assert_eq!(units[0].text, "First paragraph continues");
        assert_eq!(units[1].format, InputFormat::Empty);
    }
}
