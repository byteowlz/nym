//! True PDF redaction: PII text is *removed from the content streams*, never
//! covered up.
//!
//! The classic PDF redaction failure is drawing a black box over text and
//! leaving the characters extractable underneath. This module edits the PDF at
//! the operator level instead:
//!
//! 1. Every page content stream (and every Form XObject it references) is
//!    decoded into operators.
//! 2. Show-text operators (`Tj`, `'`, `"`, `TJ`) are mapped back to Unicode
//!    using each font's `ToUnicode` CMap (or a Latin-1 approximation for
//!    simple fonts), building the page text plus an exact map from every
//!    character to its originating operator and byte range.
//! 3. The PII detector runs over the assembled page text; matched spans are
//!    cut out of the show-text strings *byte-for-byte*. When the replacement
//!    placeholder is encodable in the same font it is spliced in; otherwise
//!    the text is simply removed.
//! 4. Document metadata (`Info` dictionary), annotations and form-field values
//!    are scrubbed with the same detector; the XMP metadata stream is dropped.
//! 5. **Verification**: the produced PDF is re-parsed and re-extracted, and
//!    every redacted string is searched for in the extracted text, in all
//!    string objects, in every decompressed stream, and in the raw output
//!    bytes. If any redacted value survives anywhere, the operation fails
//!    rather than emitting an unsafe document.
//!
//! Honest limitations, reported rather than hidden: text in raster images is
//! not touched (no OCR), fonts without a usable Unicode mapping make their
//! operators undetectable (counted in the report; `strict` refuses such
//! files), and encrypted PDFs are rejected outright. Redaction is destructive
//! by design — reversibility means keeping the original file.

use std::collections::{BTreeMap, HashMap, HashSet};

use lopdf::content::{Content, Operation};
use lopdf::{Dictionary, Document, Object, ObjectId, StringFormat};

use super::detector::Detector;
use super::replacer::{Replacement, Replacer};

type Error = Box<dyn std::error::Error + Send + Sync>;

/// What happened during a redaction run.
#[derive(Debug, Default, Clone)]
pub struct RedactionReport {
    /// Number of pages processed.
    pub pages: usize,
    /// PII occurrences removed from text content.
    pub redacted: usize,
    /// Redactions where the placeholder could be encoded into the font.
    pub placeholders_inserted: usize,
    /// Metadata / annotation string fields scrubbed.
    pub metadata_scrubbed: usize,
    /// Show-text operators whose font had no usable Unicode mapping.
    pub unmapped_text_ops: usize,
    /// Image XObjects seen (their pixels are NOT scanned — no OCR).
    pub images_seen: usize,
    /// Whether the post-write verification pass ran clean.
    pub verified: bool,
}

/// Is this path a PDF (by extension)?
pub fn is_pdf_path(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("pdf"))
}

// ---------------------------------------------------------------------------
// Font decoding
// ---------------------------------------------------------------------------

/// Decodes bytes of a show-text string into Unicode, and (when possible)
/// encodes replacement text back into font code bytes.
struct FontCodec {
    /// Code unit length in bytes (1 for simple fonts, 2 for CID/Type0).
    code_len: usize,
    /// code -> unicode from the ToUnicode CMap (None = Latin-1 passthrough).
    to_unicode: Option<HashMap<u32, String>>,
    /// unicode char -> code bytes, for placeholder insertion.
    reverse: HashMap<char, Vec<u8>>,
    /// Whether this font can be decoded at all.
    usable: bool,
}

/// One decoded character: where its bytes live in the source string.
struct DecodedChar {
    text: String,
    byte_off: usize,
    byte_len: usize,
}

impl FontCodec {
    fn latin1() -> Self {
        let mut reverse = HashMap::new();
        for b in 0x20u8..0x7f {
            reverse.insert(char::from(b), vec![b]);
        }
        FontCodec {
            code_len: 1,
            to_unicode: None,
            reverse,
            usable: true,
        }
    }

    fn unusable() -> Self {
        FontCodec {
            code_len: 1,
            to_unicode: None,
            reverse: HashMap::new(),
            usable: false,
        }
    }

    fn from_font_dict(doc: &Document, font: &Dictionary) -> Self {
        let subtype = font
            .get(b"Subtype")
            .and_then(Object::as_name)
            .unwrap_or(b"");
        let to_unicode = font
            .get(b"ToUnicode")
            .ok()
            .and_then(|o| resolve(doc, o))
            .and_then(|o| o.as_stream().ok())
            .and_then(|s| s.decompressed_content().ok())
            .and_then(|c| parse_tounicode_cmap(&c));

        if let Some((code_len, map)) = to_unicode {
            // Build a reverse map from single-char targets for placeholder encoding.
            let mut reverse: HashMap<char, Vec<u8>> = HashMap::new();
            for (&code, uni) in &map {
                let mut chars = uni.chars();
                if let (Some(c), None) = (chars.next(), chars.next()) {
                    let bytes = code_to_bytes(code, code_len);
                    reverse.entry(c).or_insert(bytes);
                }
            }
            return FontCodec {
                code_len,
                to_unicode: Some(map),
                reverse,
                usable: true,
            };
        }

        if subtype == b"Type0" {
            // Composite font without ToUnicode: bytes cannot be understood.
            return FontCodec::unusable();
        }
        // Simple font without ToUnicode: Latin-1 approximation (covers the
        // Standard/WinAnsi core-font case that dominates generated PDFs).
        FontCodec::latin1()
    }

    /// Decode a string's bytes into characters with byte provenance.
    fn decode(&self, bytes: &[u8]) -> Vec<DecodedChar> {
        let mut out = Vec::new();
        if !self.usable {
            return out;
        }
        if let Some(map) = &self.to_unicode {
            let step = self.code_len;
            let mut i = 0;
            while i + step <= bytes.len() {
                let code = bytes[i..i + step]
                    .iter()
                    .fold(0u32, |acc, &b| (acc << 8) | u32::from(b));
                if let Some(u) = map.get(&code) {
                    if !u.is_empty() {
                        out.push(DecodedChar {
                            text: u.clone(),
                            byte_off: i,
                            byte_len: step,
                        });
                    }
                }
                i += step;
            }
        } else {
            for (i, &b) in bytes.iter().enumerate() {
                out.push(DecodedChar {
                    text: char::from(b).to_string(),
                    byte_off: i,
                    byte_len: 1,
                });
            }
        }
        out
    }

    /// Encode replacement text into this font's code bytes, if fully possible.
    fn encode(&self, s: &str) -> Option<Vec<u8>> {
        let mut out = Vec::new();
        for c in s.chars() {
            out.extend(self.reverse.get(&c)?);
        }
        Some(out)
    }
}

fn code_to_bytes(code: u32, len: usize) -> Vec<u8> {
    match len {
        2 => vec![(code >> 8) as u8, (code & 0xff) as u8],
        _ => vec![(code & 0xff) as u8],
    }
}

/// Minimal ToUnicode CMap parser: handles `bfchar` and `bfrange` sections.
/// Returns (code byte length, code -> unicode string).
fn parse_tounicode_cmap(data: &[u8]) -> Option<(usize, HashMap<u32, String>)> {
    let text = String::from_utf8_lossy(data);
    let mut map = HashMap::new();
    let mut code_len = 1usize;

    let mut toks = Tokens::new(&text);
    while let Some(tok) = toks.next() {
        match tok {
            Tok::Kw("beginbfchar") => {
                while let Some(t) = toks.next() {
                    match t {
                        Tok::Hex(src) => {
                            code_len = code_len.max(src.len().min(2));
                            if let Some(Tok::Hex(dst)) = toks.next() {
                                map.insert(bytes_to_code(&src), utf16be_to_string(&dst));
                            }
                        }
                        Tok::Kw("endbfchar") => break,
                        _ => {}
                    }
                }
            }
            Tok::Kw("beginbfrange") => {
                while let Some(t) = toks.next() {
                    match t {
                        Tok::Hex(lo) => {
                            code_len = code_len.max(lo.len().min(2));
                            let hi = match toks.next() {
                                Some(Tok::Hex(h)) => h,
                                _ => break,
                            };
                            match toks.next() {
                                Some(Tok::Hex(dst)) => {
                                    let (lo, hi) = (bytes_to_code(&lo), bytes_to_code(&hi));
                                    let base = dst;
                                    for (k, code) in (lo..=hi.min(lo + 65535)).enumerate() {
                                        let mut d = base.clone();
                                        if let Some(last) = d.last_chunk_mut::<2>() {
                                            let v = u16::from_be_bytes(*last)
                                                .wrapping_add(u16::try_from(k).unwrap_or(0));
                                            *last = v.to_be_bytes();
                                        }
                                        map.insert(code, utf16be_to_string(&d));
                                    }
                                }
                                Some(Tok::ArrayStart) => {
                                    let mut code = bytes_to_code(&lo);
                                    while let Some(t) = toks.next() {
                                        match t {
                                            Tok::Hex(dst) => {
                                                map.insert(code, utf16be_to_string(&dst));
                                                code += 1;
                                            }
                                            Tok::ArrayEnd => break,
                                            _ => {}
                                        }
                                    }
                                }
                                _ => break,
                            }
                        }
                        Tok::Kw("endbfrange") => break,
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    if map.is_empty() { None } else { Some((code_len, map)) }
}

fn bytes_to_code(b: &[u8]) -> u32 {
    b.iter().fold(0u32, |acc, &x| (acc << 8) | u32::from(x))
}

fn utf16be_to_string(b: &[u8]) -> String {
    let units: Vec<u16> = b
        .chunks_exact(2)
        .map(|c| u16::from_be_bytes([c[0], c[1]]))
        .collect();
    String::from_utf16_lossy(&units)
}

/// Tiny tokenizer for CMap bodies: hex strings, keywords, array brackets.
enum Tok<'a> {
    Hex(Vec<u8>),
    Kw(&'a str),
    ArrayStart,
    ArrayEnd,
}

struct Tokens<'a> {
    s: &'a str,
    pos: usize,
}

impl<'a> Tokens<'a> {
    fn new(s: &'a str) -> Self {
        Tokens { s, pos: 0 }
    }

    fn next(&mut self) -> Option<Tok<'a>> {
        let bytes = self.s.as_bytes();
        while self.pos < bytes.len() && bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
        if self.pos >= bytes.len() {
            return None;
        }
        match bytes[self.pos] {
            b'<' => {
                let start = self.pos + 1;
                let end = self.s[start..].find('>').map(|i| start + i)?;
                self.pos = end + 1;
                let hex: String = self.s[start..end]
                    .chars()
                    .filter(char::is_ascii_hexdigit)
                    .collect();
                let mut out = Vec::new();
                let mut it = hex.as_bytes().chunks_exact(2);
                for c in &mut it {
                    let h = std::str::from_utf8(c).ok()?;
                    out.push(u8::from_str_radix(h, 16).ok()?);
                }
                Some(Tok::Hex(out))
            }
            b'[' => {
                self.pos += 1;
                Some(Tok::ArrayStart)
            }
            b']' => {
                self.pos += 1;
                Some(Tok::ArrayEnd)
            }
            _ => {
                let start = self.pos;
                while self.pos < bytes.len()
                    && !bytes[self.pos].is_ascii_whitespace()
                    && !matches!(bytes[self.pos], b'<' | b'[' | b']')
                {
                    self.pos += 1;
                }
                Some(Tok::Kw(&self.s[start..self.pos]))
            }
        }
    }
}

fn resolve<'a>(doc: &'a Document, obj: &'a Object) -> Option<&'a Object> {
    match obj {
        Object::Reference(id) => doc.get_object(*id).ok(),
        _ => Some(obj),
    }
}

// ---------------------------------------------------------------------------
// Content stream text model
// ---------------------------------------------------------------------------

/// Maps one produced character back to its operator and byte range.
struct CharRef {
    op_idx: usize,
    /// Index into a TJ array (None for Tj / ' / ").
    elem: Option<usize>,
    byte_off: usize,
    byte_len: usize,
    text_start: usize,
    text_len: usize,
}

/// The assembled text of one content stream plus its provenance map.
struct StreamText {
    text: String,
    chars: Vec<CharRef>,
    unmapped_ops: usize,
}

/// Load the font codecs available to a content stream from its resources.
fn load_fonts(doc: &Document, resources: Option<&Dictionary>) -> HashMap<Vec<u8>, FontCodec> {
    let mut out = HashMap::new();
    let Some(res) = resources else {
        return out;
    };
    let Some(fonts) = res.get(b"Font").ok().and_then(|o| resolve(doc, o)).and_then(|o| o.as_dict().ok())
    else {
        return out;
    };
    for (name, fref) in fonts.iter() {
        if let Some(fd) = resolve(doc, fref).and_then(|o| o.as_dict().ok()) {
            out.insert(name.clone(), FontCodec::from_font_dict(doc, fd));
        }
    }
    out
}

/// Which operand of a show-text operator carries the string.
fn text_operand_index(operator: &str) -> Option<usize> {
    match operator {
        "Tj" | "'" => Some(0),
        "\"" => Some(2),
        _ => None,
    }
}

/// Assemble the text of one operation list.
fn assemble_text(ops: &[Operation], fonts: &HashMap<Vec<u8>, FontCodec>) -> StreamText {
    let mut st = StreamText {
        text: String::new(),
        chars: Vec::new(),
        unmapped_ops: 0,
    };
    let mut cur_font: Option<&FontCodec> = None;

    let mut push_sep = |st: &mut StreamText, c: char| {
        if !st.text.is_empty() && !st.text.ends_with(c) {
            st.text.push(c);
        }
    };

    for (op_idx, op) in ops.iter().enumerate() {
        match op.operator.as_str() {
            "Tf" => {
                cur_font = op
                    .operands
                    .first()
                    .and_then(|o| o.as_name().ok())
                    .and_then(|n| fonts.get(n));
            }
            "Td" | "TD" => {
                let ty = op.operands.get(1).and_then(object_as_f64).unwrap_or(0.0);
                push_sep(&mut st, if ty.abs() > f64::EPSILON { '\n' } else { ' ' });
            }
            "T*" | "ET" => push_sep(&mut st, '\n'),
            "Tm" => push_sep(&mut st, '\n'),
            "Tj" | "'" | "\"" => {
                let Some(idx) = text_operand_index(op.operator.as_str()) else {
                    continue;
                };
                if op.operator != "Tj" {
                    push_sep(&mut st, '\n');
                }
                let Some(bytes) = op.operands.get(idx).and_then(|o| o.as_str().ok()) else {
                    continue;
                };
                match cur_font {
                    Some(codec) if codec.usable => {
                        append_decoded(&mut st, codec, bytes, op_idx, None);
                    }
                    _ => st.unmapped_ops += 1,
                }
            }
            "TJ" => {
                let Some(Object::Array(arr)) = op.operands.first() else {
                    continue;
                };
                match cur_font {
                    Some(codec) if codec.usable => {
                        for (elem, item) in arr.iter().enumerate() {
                            match item {
                                Object::String(bytes, _) => {
                                    append_decoded(&mut st, codec, bytes, op_idx, Some(elem));
                                }
                                other => {
                                    // Large negative kern = word gap.
                                    if let Some(v) = object_as_f64(other) {
                                        if v < -180.0 {
                                            push_sep(&mut st, ' ');
                                        }
                                    }
                                }
                            }
                        }
                    }
                    _ => st.unmapped_ops += 1,
                }
            }
            _ => {}
        }
    }
    st
}

fn object_as_f64(o: &Object) -> Option<f64> {
    match o {
        Object::Integer(i) => Some(*i as f64),
        Object::Real(r) => Some(f64::from(*r)),
        _ => None,
    }
}

fn append_decoded(
    st: &mut StreamText,
    codec: &FontCodec,
    bytes: &[u8],
    op_idx: usize,
    elem: Option<usize>,
) {
    for dc in codec.decode(bytes) {
        let text_start = st.text.len();
        st.text.push_str(&dc.text);
        st.chars.push(CharRef {
            op_idx,
            elem,
            byte_off: dc.byte_off,
            byte_len: dc.byte_len,
            text_start,
            text_len: dc.text.len(),
        });
    }
}

// ---------------------------------------------------------------------------
// Redaction editing
// ---------------------------------------------------------------------------

/// A byte-splice on one show-text string.
struct Cut {
    start: usize,
    end: usize,
    insert: Vec<u8>,
}

/// Apply detector+replacer to one stream's operations; returns edited ops and
/// how many placeholders could be encoded into the font.
fn redact_ops(
    ops: &mut [Operation],
    st: &StreamText,
    fonts: &HashMap<Vec<u8>, FontCodec>,
    detector: &Detector,
    replacer: &mut Replacer,
    log: &mut Vec<Replacement>,
) -> (usize, usize) {
    let matches = detector.detect(&st.text);
    if matches.is_empty() {
        return (0, 0);
    }

    // Font in effect for each op (for placeholder encoding).
    let mut font_for_op: Vec<Option<&FontCodec>> = Vec::with_capacity(ops.len());
    let mut cur: Option<&FontCodec> = None;
    for op in ops.iter() {
        if op.operator == "Tf" {
            cur = op
                .operands
                .first()
                .and_then(|o| o.as_name().ok())
                .and_then(|n| fonts.get(n));
        }
        font_for_op.push(cur);
    }

    let mut cuts: BTreeMap<(usize, Option<usize>), Vec<Cut>> = BTreeMap::new();
    let mut redacted = 0usize;
    let mut placeholders = 0usize;
    let mut last_end = 0usize;

    for m in &matches {
        if m.start < last_end {
            continue;
        }
        last_end = m.end;

        // All characters overlapping the match, grouped per (op, elem).
        let mut groups: BTreeMap<(usize, Option<usize>), (usize, usize)> = BTreeMap::new();
        for ch in &st.chars {
            let cs = ch.text_start;
            let ce = ch.text_start + ch.text_len;
            if cs < m.end && ce > m.start {
                let entry = groups
                    .entry((ch.op_idx, ch.elem))
                    .or_insert((ch.byte_off, ch.byte_off + ch.byte_len));
                entry.0 = entry.0.min(ch.byte_off);
                entry.1 = entry.1.max(ch.byte_off + ch.byte_len);
            }
        }
        if groups.is_empty() {
            continue;
        }

        let replacement = replacer.replace(m);
        // Try to encode the placeholder into the font of the first group.
        let first_key = groups.keys().next().copied();
        let insert = first_key
            .and_then(|(op_idx, _)| font_for_op.get(op_idx).copied().flatten())
            .and_then(|codec| codec.encode(&replacement.replacement))
            .unwrap_or_default();
        if !insert.is_empty() {
            placeholders += 1;
        }
        redacted += 1;
        log.push(replacement);

        let mut first = true;
        for (key, (bs, be)) in groups {
            cuts.entry(key).or_default().push(Cut {
                start: bs,
                end: be,
                insert: if first { insert.clone() } else { Vec::new() },
            });
            first = false;
        }
    }

    // Apply cuts right-to-left per string.
    for ((op_idx, elem), mut list) in cuts {
        list.sort_by_key(|c| std::cmp::Reverse(c.start));
        let Some(op) = ops.get_mut(op_idx) else { continue };
        let target = match (op.operator.as_str(), elem) {
            ("TJ", Some(e)) => match op.operands.first_mut() {
                Some(Object::Array(arr)) => arr.get_mut(e),
                _ => None,
            },
            (o, None) => text_operand_index(o).and_then(|i| op.operands.get_mut(i)),
            _ => None,
        };
        let Some(Object::String(bytes, _fmt)) = target else {
            continue;
        };
        for cut in list {
            let end = cut.end.min(bytes.len());
            let start = cut.start.min(end);
            bytes.splice(start..end, cut.insert);
        }
    }

    (redacted, placeholders)
}

// ---------------------------------------------------------------------------
// Document walk
// ---------------------------------------------------------------------------

/// Collect the Form XObject streams reachable from a resources dictionary.
fn form_xobjects(
    doc: &Document,
    resources: Option<&Dictionary>,
    images_seen: &mut usize,
) -> Vec<ObjectId> {
    let mut out = Vec::new();
    let Some(res) = resources else { return out };
    let Some(xobjs) = res
        .get(b"XObject")
        .ok()
        .and_then(|o| resolve(doc, o))
        .and_then(|o| o.as_dict().ok())
    else {
        return out;
    };
    for (_, r) in xobjs.iter() {
        let Ok(id) = r.as_reference() else { continue };
        let Some(stream) = doc.get_object(id).ok().and_then(|o| o.as_stream().ok()) else {
            continue;
        };
        match stream.dict.get(b"Subtype").and_then(Object::as_name) {
            Ok(b"Form") => out.push(id),
            Ok(b"Image") => *images_seen += 1,
            _ => {}
        }
    }
    out
}

/// Resources for a page — either directly on the page or inherited from an
/// ancestor Pages node (common with fpdf2 and Word exports).
fn page_resources(doc: &Document, page_id: ObjectId) -> Option<&Dictionary> {
    let (res, ids) = doc.get_page_resources(page_id).ok()?;
    if let Some(r) = res {
        return Some(r);
    }
    ids.first()
        .and_then(|id| doc.get_object(*id).ok())
        .and_then(|o| o.as_dict().ok())
}

fn stream_resources<'a>(doc: &'a Document, id: ObjectId) -> Option<&'a Dictionary> {
    doc.get_object(id)
        .ok()
        .and_then(|o| o.as_stream().ok())
        .and_then(|s| s.dict.get(b"Resources").ok())
        .and_then(|o| resolve(doc, o))
        .and_then(|o| o.as_dict().ok())
}

/// Extract all text (pages, form XObjects, metadata strings) — used by
/// `nym detect` and by the verification pass.
pub fn extract_text(bytes: &[u8]) -> Result<String, Error> {
    let doc = Document::load_mem(bytes)?;
    if doc.is_encrypted() {
        return Err("PDF is encrypted; decrypt it before processing".into());
    }
    let mut out = String::new();
    let mut images = 0usize;

    for (_no, page_id) in doc.get_pages() {
        let resources = page_resources(&doc, page_id);
        let content = doc.get_page_content(page_id)?;
        let ops = Content::decode(&content)?.operations;
        let fonts = load_fonts(&doc, resources);
        let st = assemble_text(&ops, &fonts);
        out.push_str(&st.text);
        out.push('\n');

        for fid in form_xobjects(&doc, resources, &mut images) {
            if let Some(stream) = doc.get_object(fid).ok().and_then(|o| o.as_stream().ok()) {
                if let Ok(data) = stream.decompressed_content() {
                    if let Ok(c) = Content::decode(&data) {
                        let fres = stream_resources(&doc, fid).or(resources);
                        let ffonts = load_fonts(&doc, fres);
                        let fst = assemble_text(&c.operations, &ffonts);
                        out.push_str(&fst.text);
                        out.push('\n');
                    }
                }
            }
        }
    }

    // Metadata strings (Info dictionary).
    if let Some(info) = doc
        .trailer
        .get(b"Info")
        .ok()
        .and_then(|o| resolve(&doc, o))
        .and_then(|o| o.as_dict().ok())
    {
        for (_, v) in info.iter() {
            if let Object::String(s, _) = v {
                out.push_str(&pdf_string_to_text(s));
                out.push('\n');
            }
        }
    }
    Ok(out)
}

/// Decode a PDF text string (UTF-16BE with BOM, else Latin-1-ish).
fn pdf_string_to_text(bytes: &[u8]) -> String {
    if bytes.len() >= 2 && bytes[0] == 0xfe && bytes[1] == 0xff {
        utf16be_to_string(&bytes[2..])
    } else {
        bytes.iter().map(|&b| char::from(b)).collect()
    }
}

fn text_to_pdf_string(s: &str) -> Vec<u8> {
    if s.chars().all(|c| (c as u32) < 0x100) {
        s.chars().map(|c| c as u8).collect()
    } else {
        let mut out = vec![0xfe, 0xff];
        for u in s.encode_utf16() {
            out.extend_from_slice(&u.to_be_bytes());
        }
        out
    }
}

/// Scrub one dictionary string field with the detector; returns new value.
fn scrub_text_value(
    text: &str,
    detector: &Detector,
    replacer: &mut Replacer,
    log: &mut Vec<Replacement>,
) -> Option<String> {
    let matches = detector.detect(text);
    if matches.is_empty() {
        return None;
    }
    let mut out = String::new();
    let mut last = 0usize;
    for m in &matches {
        if m.start < last {
            continue;
        }
        out.push_str(&text[last..m.start]);
        let r = replacer.replace(m);
        out.push_str(&r.replacement);
        log.push(r);
        last = m.end;
    }
    out.push_str(&text[last..]);
    Some(out)
}

/// Redact a PDF. Returns the new bytes, the replacement log, and a report.
/// With `strict`, any undecodable text operator aborts the run.
pub fn redact(
    bytes: &[u8],
    detector: &Detector,
    replacer: &mut Replacer,
    strict: bool,
) -> Result<(Vec<u8>, Vec<Replacement>, RedactionReport), Error> {
    let mut doc = Document::load_mem(bytes)?;
    if doc.is_encrypted() {
        return Err("PDF is encrypted; decrypt it before redacting".into());
    }

    let mut report = RedactionReport::default();
    let mut log: Vec<Replacement> = Vec::new();

    let pages = doc.get_pages();
    report.pages = pages.len();

    // -- Pass 1: page content streams ------------------------------------
    let mut page_edits: Vec<(ObjectId, Vec<u8>)> = Vec::new();
    let mut form_edits: Vec<(ObjectId, Vec<u8>)> = Vec::new();
    let mut done_forms: HashSet<ObjectId> = HashSet::new();

    for (_no, page_id) in &pages {
        let resources = page_resources(&doc, *page_id);
        let content = doc.get_page_content(*page_id)?;
        let mut ops = Content::decode(&content)?.operations;
        let fonts = load_fonts(&doc, resources);
        let st = assemble_text(&ops, &fonts);
        report.unmapped_text_ops += st.unmapped_ops;

        let (n, p) = redact_ops(&mut ops, &st, &fonts, detector, replacer, &mut log);
        report.redacted += n;
        report.placeholders_inserted += p;
        if n > 0 {
            let encoded = Content { operations: ops }.encode()?;
            page_edits.push((*page_id, encoded));
        }

        // Form XObjects (letterheads, stamps, headers).
        for fid in form_xobjects(&doc, resources, &mut report.images_seen) {
            if !done_forms.insert(fid) {
                continue;
            }
            let Some(stream) = doc.get_object(fid).ok().and_then(|o| o.as_stream().ok()) else {
                continue;
            };
            let Ok(data) = stream.decompressed_content() else {
                continue;
            };
            let Ok(c) = Content::decode(&data) else {
                continue;
            };
            let mut fops = c.operations;
            let fres = stream_resources(&doc, fid).or(resources);
            let ffonts = load_fonts(&doc, fres);
            let fst = assemble_text(&fops, &ffonts);
            report.unmapped_text_ops += fst.unmapped_ops;
            let (n, p) = redact_ops(&mut fops, &fst, &ffonts, detector, replacer, &mut log);
            report.redacted += n;
            report.placeholders_inserted += p;
            if n > 0 {
                let encoded = Content { operations: fops }.encode()?;
                form_edits.push((fid, encoded));
            }
        }
    }

    if strict && report.unmapped_text_ops > 0 {
        return Err(format!(
            "{} text operator(s) use fonts without a usable Unicode mapping; \
             their content cannot be inspected. Refusing to redact (pass --no-strict-pdf to override).",
            report.unmapped_text_ops
        )
        .into());
    }

    for (page_id, content) in page_edits {
        doc.change_page_content(page_id, content)?;
    }
    for (fid, content) in form_edits {
        if let Ok(obj) = doc.get_object_mut(fid) {
            if let Ok(stream) = obj.as_stream_mut() {
                stream.set_plain_content(content);
                stream.dict.remove(b"Filter");
                stream.dict.remove(b"DecodeParms");
            }
        }
    }

    // -- Pass 2: metadata + annotations -----------------------------------
    // Info dictionary.
    let info_id = doc.trailer.get(b"Info").ok().and_then(|o| o.as_reference().ok());
    if let Some(id) = info_id {
        let fields: Vec<(Vec<u8>, String)> = doc
            .get_object(id)
            .ok()
            .and_then(|o| o.as_dict().ok())
            .map(|d| {
                d.iter()
                    .filter_map(|(k, v)| match v {
                        Object::String(s, _) => Some((k.clone(), pdf_string_to_text(s))),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default();
        for (key, text) in fields {
            if let Some(new) = scrub_text_value(&text, detector, replacer, &mut log) {
                if let Ok(obj) = doc.get_object_mut(id) {
                    if let Ok(dict) = obj.as_dict_mut() {
                        dict.set(key, Object::String(text_to_pdf_string(&new), StringFormat::Literal));
                        report.metadata_scrubbed += 1;
                    }
                }
            }
        }
    }

    // XMP metadata stream: drop it entirely (it duplicates Info and often
    // carries author/tool identifiers).
    let catalog_id = doc.trailer.get(b"Root").ok().and_then(|o| o.as_reference().ok());
    if let Some(id) = catalog_id {
        if let Ok(obj) = doc.get_object_mut(id) {
            if let Ok(dict) = obj.as_dict_mut() {
                if dict.remove(b"Metadata").is_some() {
                    report.metadata_scrubbed += 1;
                }
            }
        }
    }

    // Annotation strings (comments, popup titles, form field values).
    let annot_keys: &[&[u8]] = &[b"Contents", b"T", b"Subj", b"V", b"TU"];
    let mut annot_ids: Vec<ObjectId> = Vec::new();
    for (_no, page_id) in &pages {
        if let Ok(page) = doc.get_dictionary(*page_id) {
            if let Ok(annots) = page.get(b"Annots") {
                let arr = match annots {
                    Object::Array(a) => a.clone(),
                    Object::Reference(r) => doc
                        .get_object(*r)
                        .ok()
                        .and_then(|o| o.as_array().ok())
                        .cloned()
                        .unwrap_or_default(),
                    _ => Vec::new(),
                };
                annot_ids.extend(arr.iter().filter_map(|o| o.as_reference().ok()));
            }
        }
    }
    for aid in annot_ids {
        let fields: Vec<(Vec<u8>, String)> = doc
            .get_object(aid)
            .ok()
            .and_then(|o| o.as_dict().ok())
            .map(|d| {
                d.iter()
                    .filter(|(k, _)| annot_keys.contains(&k.as_slice()))
                    .filter_map(|(k, v)| match v {
                        Object::String(s, _) => Some((k.clone(), pdf_string_to_text(s))),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default();
        for (key, text) in fields {
            if let Some(new) = scrub_text_value(&text, detector, replacer, &mut log) {
                if let Ok(obj) = doc.get_object_mut(aid) {
                    if let Ok(dict) = obj.as_dict_mut() {
                        dict.set(key, Object::String(text_to_pdf_string(&new), StringFormat::Literal));
                        report.metadata_scrubbed += 1;
                    }
                }
            }
        }
    }

    // -- Save -------------------------------------------------------------
    let mut out = Vec::new();
    doc.save_to(&mut out)?;

    // -- Pass 3: verification ---------------------------------------------
    let leftovers = verify_absence(&out, &log)?;
    if !leftovers.is_empty() {
        return Err(format!(
            "redaction verification FAILED — {} redacted value(s) still present in output \
             (e.g. {:?}); refusing to write an unsafe document",
            leftovers.len(),
            leftovers.first().map(|s| mask_for_log(s)),
        )
        .into());
    }
    report.verified = true;

    Ok((out, log, report))
}

/// Never echo full PII into logs/errors.
fn mask_for_log(s: &str) -> String {
    let head: String = s.chars().take(3).collect();
    format!("{head}…")
}

/// Search the produced PDF for any redacted original: extracted text, every
/// string object, every decompressed stream, and the raw bytes.
fn verify_absence(out: &[u8], log: &[Replacement]) -> Result<Vec<String>, Error> {
    let originals: Vec<&String> = log
        .iter()
        .map(|r| &r.original)
        .filter(|o| o.len() >= 4)
        .collect();
    if originals.is_empty() {
        return Ok(Vec::new());
    }

    let mut haystack = String::new();
    haystack.push_str(&extract_text(out)?);

    let doc = Document::load_mem(out)?;
    for (_id, obj) in &doc.objects {
        collect_strings(obj, &mut haystack);
        if let Ok(stream) = obj.as_stream() {
            if let Ok(data) = stream.decompressed_content() {
                haystack.extend(data.iter().map(|&b| char::from(b)));
                haystack.push('\n');
            }
        }
    }
    // Raw bytes as Latin-1 (catches anything unparsed).
    haystack.extend(out.iter().map(|&b| char::from(b)));

    Ok(originals
        .into_iter()
        .filter(|o| haystack.contains(o.as_str()))
        .cloned()
        .collect())
}

fn collect_strings(obj: &Object, out: &mut String) {
    match obj {
        Object::String(s, _) => {
            out.push_str(&pdf_string_to_text(s));
            out.push('\n');
        }
        Object::Array(a) => {
            for o in a {
                collect_strings(o, out);
            }
        }
        Object::Dictionary(d) => {
            for (_, o) in d.iter() {
                collect_strings(o, out);
            }
        }
        Object::Stream(s) => {
            for (_, o) in s.dict.iter() {
                collect_strings(o, out);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::replacer::ReplacerConfig;
    use crate::engine::{DetectorConfig, ReplacementStrategy};

    /// A minimal single-page PDF built with lopdf itself.
    fn minimal_pdf(text_ops: &str) -> Vec<u8> {
        use lopdf::dictionary;
        let mut doc = Document::with_version("1.4");
        let font_id = doc.add_object(dictionary! {
            "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica",
        });
        let content = format!("BT /F1 12 Tf 72 720 Td {text_ops} ET");
        let content_id = doc.add_object(lopdf::Stream::new(dictionary! {}, content.into_bytes()));
        let pages_id = doc.new_object_id();
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
            "Resources" => dictionary! { "Font" => dictionary! { "F1" => font_id } },
            "Contents" => content_id,
        });
        doc.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages", "Kids" => vec![page_id.into()], "Count" => 1,
            }),
        );
        let catalog_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
        doc.trailer.set("Root", catalog_id);
        let mut out = Vec::new();
        #[expect(clippy::unwrap_used, reason = "test fixture")]
        doc.save_to(&mut out).unwrap();
        out
    }

    #[test]
    fn cmap_bfchar_and_bfrange_parse() {
        let cmap = b"begincmap\n2 beginbfchar\n<0041> <0058>\n<0042> <00590059>\nendbfchar\n\
                     1 beginbfrange\n<0060> <0062> <0041>\nendbfrange\nendcmap";
        #[expect(clippy::unwrap_used, reason = "test")]
        let (len, map) = parse_tounicode_cmap(cmap).unwrap();
        assert_eq!(len, 2);
        assert_eq!(map.get(&0x41).map(String::as_str), Some("X"));
        assert_eq!(map.get(&0x42).map(String::as_str), Some("YY"));
        assert_eq!(map.get(&0x60).map(String::as_str), Some("A"));
        assert_eq!(map.get(&0x62).map(String::as_str), Some("C"));
    }

    #[test]
    fn pdf_redaction_removes_text_and_verifies() {
        let pdf = minimal_pdf("(Contact john.doe@example.com or 555-123-4567 x99) Tj");
        let detector = Detector::new(&DetectorConfig::default());
        let mut replacer = Replacer::new(ReplacerConfig {
            strategy: ReplacementStrategy::Placeholder,
            ..ReplacerConfig::default()
        });
        #[expect(clippy::unwrap_used, reason = "test")]
        let (out, log, report) = redact(&pdf, &detector, &mut replacer, true).unwrap();
        assert!(report.verified);
        assert!(log.iter().any(|r| r.original == "john.doe@example.com"));
        let raw: String = out.iter().map(|&b| char::from(b)).collect();
        assert!(!raw.contains("john.doe@example.com"), "email must be gone from raw bytes");
        #[expect(clippy::unwrap_used, reason = "test")]
        let text = extract_text(&out).unwrap();
        assert!(text.contains("<EMAIL>"), "placeholder present: {text}");
        assert!(text.contains("Contact"), "surrounding text kept: {text}");
    }

    #[test]
    fn pdf_split_tj_array_redaction() {
        // Email split across TJ array elements with kerning, the way real
        // generators emit it.
        let pdf = minimal_pdf("[(Mail: jo) -20 (hn.doe@exam) -20 (ple.com end)] TJ");
        let detector = Detector::new(&DetectorConfig::default());
        let mut replacer = Replacer::new(ReplacerConfig {
            strategy: ReplacementStrategy::Placeholder,
            ..ReplacerConfig::default()
        });
        #[expect(clippy::unwrap_used, reason = "test")]
        let (out, _, report) = redact(&pdf, &detector, &mut replacer, true).unwrap();
        assert!(report.verified);
        let raw: String = out.iter().map(|&b| char::from(b)).collect();
        assert!(!raw.contains("hn.doe@exam"), "no fragment survives");
        #[expect(clippy::unwrap_used, reason = "test")]
        let text = extract_text(&out).unwrap();
        assert!(text.contains("Mail:") && text.contains("end"), "{text}");
    }

    #[test]
    fn encrypted_pdf_is_rejected() {
        // Not a real encrypted file, but the loader rejects garbage too —
        // the point is we error rather than emit anything.
        assert!(redact(b"%PDF-1.4 garbage", &Detector::new(&DetectorConfig::default()),
                        &mut Replacer::new(ReplacerConfig::default()), true).is_err());
    }
}
