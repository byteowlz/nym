//! Bounded, replay-safe resolution of textual input formats.
use anyhow::Result;
use clap::ValueEnum;
use std::{
    io::{self, BufRead, BufReader, Read},
    path::Path,
};

const SNIFF_BYTES: u64 = 64 * 1024;

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum FormatArg {
    /// Plain text; bypass structured sniffing.
    Text,
    /// One JSON document; scans decoded string values.
    Json,
    /// Newline-delimited JSON; scans each record's decoded string values.
    #[value(alias = "ndjson")]
    Jsonl,
}

pub type TextReader = Box<dyn BufRead + Send>;

pub fn open(path: Option<&Path>, explicit: Option<FormatArg>) -> Result<(FormatArg, TextReader)> {
    let reader: Box<dyn Read + Send> = match path {
        Some(path) => Box::new(std::fs::File::open(path)?),
        None => Box::new(io::stdin()),
    };
    resolve(reader, path, explicit)
}

fn resolve(
    reader: Box<dyn Read + Send>,
    path: Option<&Path>,
    explicit: Option<FormatArg>,
) -> Result<(FormatArg, TextReader)> {
    if let Some(format) = explicit.or_else(|| extension(path)) {
        return Ok((format, Box::new(BufReader::new(reader))));
    }
    let mut reader = BufReader::new(reader);
    let mut prefix = Vec::new();
    loop {
        let remaining = SNIFF_BYTES - prefix.len() as u64;
        if remaining == 0
            || reader
                .by_ref()
                .take(remaining)
                .read_until(b'\n', &mut prefix)?
                == 0
        {
            break;
        }
        let text = String::from_utf8_lossy(&prefix);
        let text = structured_text(&text);
        if text.is_empty() {
            continue;
        }
        if !text.starts_with(['{', '[']) || sniff(&prefix) == FormatArg::Text {
            break;
        }
        if sniff(&prefix) == FormatArg::Jsonl {
            break;
        }
        // A complete first JSON document needs another nonblank line/EOF to
        // distinguish singleton JSON from records. Incomplete pretty JSON
        // continues only up to the fixed sniff budget.
    }
    let format = sniff(&prefix);
    // Replay every byte, even an incomplete UTF-8 sequence at the boundary.
    Ok((
        format,
        Box::new(BufReader::new(io::Cursor::new(prefix).chain(reader))),
    ))
}

fn extension(path: Option<&Path>) -> Option<FormatArg> {
    let ext = path?.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "json" => Some(FormatArg::Json),
        "jsonl" | "ndjson" => Some(FormatArg::Jsonl),
        // Named non-structured formats remain text; sniff only extensionless/stdin.
        "" => None,
        _ => Some(FormatArg::Text),
    }
}

pub fn structured_text(text: &str) -> &str {
    text.strip_prefix('\u{feff}').unwrap_or(text).trim_start()
}

fn sniff(prefix: &[u8]) -> FormatArg {
    // A UTF-8 character can straddle the fixed byte limit.
    let text = match std::str::from_utf8(prefix) {
        Ok(text) => text,
        Err(err) => std::str::from_utf8(&prefix[..err.valid_up_to()]).unwrap_or(""),
    };
    let text = structured_text(text);
    if !text.starts_with(['{', '[']) {
        return FormatArg::Text;
    }
    let mut values = serde_json::Deserializer::from_str(text).into_iter::<serde_json::Value>();
    match values.next() {
        Some(Ok(_)) => {
            let rest = text[values.byte_offset()..].trim();
            if rest.is_empty() {
                FormatArg::Json
            } else if text
                .lines()
                .next()
                .is_some_and(|line| serde_json::from_str::<serde_json::Value>(line).is_ok())
            {
                FormatArg::Jsonl
            } else {
                FormatArg::Json
            }
        }
        Some(Err(err)) if err.is_eof() => {
            // Only JSON-looking incomplete objects/arrays. Code such as {foo: x}
            // or [some_expression] is deliberately not declared JSON.
            let after = text[1..].trim_start();
            if after.is_empty()
                || after.starts_with([
                    '"', '{', '[', '}', ']', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                ])
                || ["true", "false", "null"]
                    .iter()
                    .any(|word| after.starts_with(word))
            {
                FormatArg::Json
            } else {
                FormatArg::Text
            }
        }
        _ => FormatArg::Text,
    }
}

/// Bridge the replay-safe blocking input into the existing async pipelines.
/// The multithreaded runtime releases its worker while stdin/file reads block.
#[cfg(feature = "streaming")]
pub struct BlockingReader(pub TextReader);

#[cfg(feature = "streaming")]
impl tokio::io::AsyncRead for BlockingReader {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<io::Result<()>> {
        let result = tokio::task::block_in_place(|| self.0.read(buf.initialize_unfilled()));
        std::task::Poll::Ready(result.map(|count| buf.advance(count)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prefix_is_bounded_and_replayed_exactly() {
        let bytes = vec![b'x'; SNIFF_BYTES as usize * 3];
        let (format, mut reader) =
            resolve(Box::new(io::Cursor::new(bytes.clone())), None, None).unwrap();
        let mut replay = Vec::new();
        reader.read_to_end(&mut replay).unwrap();
        assert_eq!((format, replay), (FormatArg::Text, bytes));
    }
    #[test]
    fn conservative_structured_sniffing() {
        for (text, expected) in [
            ("{\"a\":1}\n{\"a\":2}\n", FormatArg::Jsonl),
            ("\u{feff} \n{\n\"a\":1\n}", FormatArg::Json),
            ("{not: JSON}", FormatArg::Text),
            ("[some_expression]", FormatArg::Text),
            ("{\"a\":1}\n{bad: record}", FormatArg::Jsonl),
        ] {
            assert_eq!(sniff(text.as_bytes()), expected);
        }
    }
    #[test]
    fn explicit_and_case_insensitive_extension_win() {
        let (format, _) = resolve(
            Box::new(io::Cursor::new(b"not json")),
            Some(Path::new("trace.NDJSON")),
            None,
        )
        .unwrap();
        assert_eq!(format, FormatArg::Jsonl);
        let (format, _) = resolve(
            Box::new(io::Cursor::new(b"not json")),
            Some(Path::new("trace.JSONL")),
            Some(FormatArg::Text),
        )
        .unwrap();
        assert_eq!(format, FormatArg::Text);
    }
}
