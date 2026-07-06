//! In-place PII redaction for office documents (docx / xlsx / pptx / ODF).
//!
//! These formats are ZIP archives of XML. Redaction rewrites only the *text
//! nodes* of the text-bearing XML parts and leaves everything else (styles,
//! layout, images, relationships) byte-identical, so formatting survives and
//! different-length replacements are safe.
//!
//! Word (and PowerPoint) fragment text across runs mid-word — an email address
//! may be split over three `<w:t>` nodes. Detection therefore runs on the
//! *paragraph* level: all text nodes inside a paragraph (`w:p` / `a:p` / `si` /
//! ODF `text:p`) are concatenated, the detector runs once over the joined text,
//! and replacement spans are mapped back onto the individual nodes. A
//! replacement that crosses node boundaries is written into the node where it
//! starts; the covered remainder is removed from the following nodes, so run
//! formatting is preserved everywhere else.
//!
//! Beyond the main body, the adapter also covers the places PII likes to hide:
//! headers/footers, footnotes/endnotes and comments (docx), speaker-notes and
//! comments (pptx), shared strings, inline strings and cell comments (xlsx —
//! formulas and numeric cell values are never touched), and the document
//! metadata (`docProps/core.xml` / ODF `meta.xml`: author, lastModifiedBy, …).

use std::collections::HashMap;
use std::io::{Cursor, Read, Write};
use std::path::Path;

use quick_xml::Reader;
use quick_xml::Writer;
use quick_xml::events::{BytesText, Event};
use zip::ZipArchive;
use zip::ZipWriter;
use zip::write::SimpleFileOptions;

use super::detector::Detector;
use super::replacer::{Replacement, Replacer};

/// Errors are surfaced as boxed trait objects, matching the other engine modules.
type Error = Box<dyn std::error::Error + Send + Sync>;

/// A replacement span in paragraph-text coordinates: `(start, end, replacement)`.
type Span = (usize, usize, String);

/// Supported office document families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OfficeFormat {
    /// Word (`.docx`)
    Docx,
    /// Excel (`.xlsx`)
    Xlsx,
    /// PowerPoint (`.pptx`)
    Pptx,
    /// OpenDocument (`.odt`, `.ods`, `.odp`)
    Odf,
}

/// Detect an office format from a file extension.
pub fn sniff_path(path: &Path) -> Option<OfficeFormat> {
    match path
        .extension()?
        .to_str()?
        .to_ascii_lowercase()
        .as_str()
    {
        "docx" => Some(OfficeFormat::Docx),
        "xlsx" => Some(OfficeFormat::Xlsx),
        "pptx" => Some(OfficeFormat::Pptx),
        "odt" | "ods" | "odp" => Some(OfficeFormat::Odf),
        _ => None,
    }
}

/// Which character data inside a paragraph group counts as document text.
#[derive(Clone, Copy)]
enum TextRule {
    /// Only text directly inside elements with these local names (`w:t`, `a:t`, `t`).
    Elems(&'static [&'static str]),
    /// All character data inside the group (ODF mixed content).
    AllInGroup,
}

/// How to process one XML part of the archive.
#[derive(Clone, Copy)]
struct PartProfile {
    /// Local names of paragraph-level grouping elements.
    groups: &'static [&'static str],
    /// Which text nodes inside a group are document text.
    rule: TextRule,
    /// Whether to manage `xml:space="preserve"` on modified text elements.
    preserve_space: bool,
}

/// Metadata elements (OOXML core properties and ODF meta) that carry free text.
const META_ELEMS: &[&str] = &[
    "creator",
    "lastModifiedBy",
    "title",
    "subject",
    "description",
    "keywords",
    "category",
    "manager",
    "company",
    "initial-creator",
];

const META_PROFILE: PartProfile = PartProfile {
    groups: META_ELEMS,
    rule: TextRule::AllInGroup,
    preserve_space: false,
};

/// Decide whether (and how) an archive entry gets its text rewritten.
fn profile_for(fmt: OfficeFormat, name: &str) -> Option<PartProfile> {
    if name == "docProps/core.xml" || name == "docProps/app.xml" || name == "meta.xml" {
        return Some(META_PROFILE);
    }
    match fmt {
        OfficeFormat::Docx => {
            let is_text_part = name == "word/document.xml"
                || name == "word/footnotes.xml"
                || name == "word/endnotes.xml"
                || name == "word/comments.xml"
                || (name.starts_with("word/header") && name.ends_with(".xml"))
                || (name.starts_with("word/footer") && name.ends_with(".xml"));
            is_text_part.then_some(PartProfile {
                groups: &["p"],
                rule: TextRule::Elems(&["t"]),
                preserve_space: true,
            })
        }
        OfficeFormat::Pptx => {
            let is_text_part = (name.starts_with("ppt/slides/")
                || name.starts_with("ppt/notesSlides/")
                || name.starts_with("ppt/comments"))
                && name.ends_with(".xml");
            is_text_part.then_some(PartProfile {
                groups: &["p"],
                rule: TextRule::Elems(&["t"]),
                preserve_space: true,
            })
        }
        OfficeFormat::Xlsx => {
            if name == "xl/sharedStrings.xml" {
                Some(PartProfile {
                    groups: &["si"],
                    rule: TextRule::Elems(&["t"]),
                    preserve_space: true,
                })
            } else if name.starts_with("xl/worksheets/sheet") && name.ends_with(".xml") {
                // Only inline strings (`<is><t>`): formulas (`f`) and cell
                // values (`v`, numbers / shared-string indices) stay untouched.
                Some(PartProfile {
                    groups: &["is"],
                    rule: TextRule::Elems(&["t"]),
                    preserve_space: true,
                })
            } else if name.starts_with("xl/comments") && name.ends_with(".xml") {
                Some(PartProfile {
                    groups: &["text"],
                    rule: TextRule::Elems(&["t"]),
                    preserve_space: true,
                })
            } else {
                None
            }
        }
        OfficeFormat::Odf => {
            let is_text_part = name == "content.xml" || name == "styles.xml";
            is_text_part.then_some(PartProfile {
                groups: &["p", "h"],
                rule: TextRule::AllInGroup,
                preserve_space: false,
            })
        }
    }
}

/// Strip the namespace prefix from a qualified name.
fn local_name(qname: &[u8]) -> &[u8] {
    qname
        .iter()
        .rposition(|&b| b == b':')
        .map_or(qname, |i| &qname[i + 1..])
}

fn is_group(profile: &PartProfile, qname: &[u8]) -> bool {
    let ln = local_name(qname);
    profile.groups.iter().any(|g| g.as_bytes() == ln)
}

/// Is character data at the current element position document text?
fn in_text_context(profile: &PartProfile, elem_stack: &[Vec<u8>]) -> bool {
    match profile.rule {
        TextRule::Elems(elems) => elem_stack.last().is_some_and(|top| {
            let ln = local_name(top);
            elems.iter().any(|t| t.as_bytes() == ln)
        }),
        TextRule::AllInGroup => true,
    }
}

/// A text node collected inside a paragraph buffer.
struct TextNode {
    /// Index of the `Event::Text` in the paragraph event buffer.
    ev_idx: usize,
    /// Index of the enclosing element's start event (for `xml:space` fixes).
    start_ev_idx: Option<usize>,
    /// Unescaped text content.
    text: String,
}

/// Rewrite one XML part: buffer each paragraph group, hand the joined text to
/// `f`, and map any returned replacement spans back onto the text nodes.
fn process_part(
    xml: &[u8],
    profile: &PartProfile,
    f: &mut dyn FnMut(&str) -> Option<Vec<Span>>,
) -> Result<Vec<u8>, Error> {
    let mut reader = Reader::from_reader(xml);
    reader.config_mut().check_end_names = false;
    let mut writer = Writer::new(Vec::with_capacity(xml.len() + 256));
    let mut buf = Vec::new();

    // Paragraph buffering state.
    let mut para: Vec<Event<'static>> = Vec::new();
    let mut group_depth = 0usize;
    let mut elem_stack: Vec<Vec<u8>> = Vec::new();
    let mut nodes: Vec<TextNode> = Vec::new();
    let mut start_stack: Vec<Option<usize>> = Vec::new();

    loop {
        let ev = reader.read_event_into(&mut buf)?;
        if matches!(ev, Event::Eof) {
            break;
        }

        if group_depth == 0 {
            if let Event::Start(ref e) = ev {
                if is_group(profile, e.name().as_ref()) {
                    group_depth = 1;
                    para.clear();
                    nodes.clear();
                    elem_stack.clear();
                    start_stack.clear();
                    para.push(ev.into_owned());
                    buf.clear();
                    continue;
                }
            }
            writer.write_event(ev)?;
            buf.clear();
            continue;
        }

        // Inside a paragraph group: buffer events and track structure.
        match &ev {
            Event::Start(e) => {
                let name = e.name().as_ref().to_vec();
                if is_group(profile, &name) {
                    group_depth += 1;
                }
                elem_stack.push(name);
                start_stack.push(Some(para.len()));
            }
            Event::End(e) => {
                elem_stack.pop();
                start_stack.pop();
                if is_group(profile, e.name().as_ref()) {
                    group_depth -= 1;
                    if group_depth == 0 {
                        para.push(ev.into_owned());
                        flush_paragraph(&mut writer, profile, &para, &nodes, f)?;
                        buf.clear();
                        continue;
                    }
                }
            }
            Event::Text(e) => {
                if in_text_context(profile, &elem_stack) {
                    let raw = std::str::from_utf8(e.as_ref())?;
                    let text = quick_xml::escape::unescape(raw)?.into_owned();
                    nodes.push(TextNode {
                        ev_idx: para.len(),
                        start_ev_idx: start_stack.last().copied().flatten(),
                        text,
                    });
                }
            }
            // quick-xml emits entity references (`&lt;` &c.) as separate events;
            // resolve them so they are part of the paragraph text. Unresolvable
            // custom entities pass through untouched (and stay invisible to
            // detection, which is the safe direction).
            Event::GeneralRef(e) => {
                if in_text_context(profile, &elem_stack) {
                    let resolved = match e.decode()?.as_ref() {
                        "lt" => Some('<'),
                        "gt" => Some('>'),
                        "amp" => Some('&'),
                        "apos" => Some('\''),
                        "quot" => Some('"'),
                        _ => e.resolve_char_ref()?,
                    };
                    if let Some(ch) = resolved {
                        nodes.push(TextNode {
                            ev_idx: para.len(),
                            start_ev_idx: start_stack.last().copied().flatten(),
                            text: ch.to_string(),
                        });
                    }
                }
            }
            _ => {}
        }
        para.push(ev.into_owned());
        buf.clear();
    }

    Ok(writer.into_inner())
}

/// Emit a buffered paragraph, applying replacement spans to its text nodes.
fn flush_paragraph(
    writer: &mut Writer<Vec<u8>>,
    profile: &PartProfile,
    para: &[Event<'static>],
    nodes: &[TextNode],
    f: &mut dyn FnMut(&str) -> Option<Vec<Span>>,
) -> Result<(), Error> {
    // Join node texts and record each node's [offset, offset+len) range.
    let mut full = String::new();
    let mut ranges = Vec::with_capacity(nodes.len());
    for n in nodes {
        let start = full.len();
        full.push_str(&n.text);
        ranges.push((start, full.len()));
    }

    let spans = if full.is_empty() { None } else { f(&full) };
    let Some(spans) = spans.filter(|s| !s.is_empty()) else {
        for ev in para {
            writer.write_event(ev.clone())?;
        }
        return Ok(());
    };

    // Compute the new text for every node.
    let mut new_texts: HashMap<usize, String> = HashMap::new();
    for (i, &(a, b)) in ranges.iter().enumerate() {
        let touched = spans.iter().any(|&(s, e, _)| s < b && e > a);
        if !touched {
            continue;
        }
        let mut out = String::new();
        let mut pos = a;
        for (s, e, repl) in &spans {
            let seg_s = (*s).max(a);
            let seg_e = (*e).min(b);
            if seg_s >= seg_e && !(*s >= a && *s < b && s == e) {
                continue;
            }
            if seg_s > pos {
                out.push_str(&full[pos..seg_s]);
            }
            // The replacement is emitted in the node where the span starts.
            if *s >= a && *s < b {
                out.push_str(repl);
            }
            pos = seg_e.max(pos);
        }
        if pos < b {
            out.push_str(&full[pos..b]);
        }
        new_texts.insert(nodes[i].ev_idx, out);
    }

    // Start-events that need an xml:space="preserve" attribute added.
    let mut needs_preserve: Vec<usize> = Vec::new();
    if profile.preserve_space {
        for n in nodes {
            if let (Some(start_idx), Some(new)) = (n.start_ev_idx, new_texts.get(&n.ev_idx)) {
                let boundary_ws = new.starts_with(char::is_whitespace)
                    || new.ends_with(char::is_whitespace);
                if boundary_ws && !new.is_empty() {
                    needs_preserve.push(start_idx);
                }
            }
        }
    }

    for (idx, ev) in para.iter().enumerate() {
        if let Some(new) = new_texts.get(&idx) {
            writer.write_event(Event::Text(BytesText::new(new)))?;
            continue;
        }
        if needs_preserve.contains(&idx) {
            if let Event::Start(e) = ev {
                let has_attr = e
                    .attributes()
                    .flatten()
                    .any(|a| a.key.as_ref() == b"xml:space");
                if !has_attr {
                    let mut e2 = e.clone().into_owned();
                    e2.push_attribute(("xml:space", "preserve"));
                    writer.write_event(Event::Start(e2))?;
                    continue;
                }
            }
        }
        writer.write_event(ev.clone())?;
    }
    Ok(())
}

/// Rewrite an office archive: parts with a profile get their paragraph text run
/// through `f`; every other entry is copied through raw (byte-identical).
fn rewrite_archive(
    bytes: &[u8],
    fmt: OfficeFormat,
    f: &mut dyn FnMut(&str) -> Option<Vec<Span>>,
) -> Result<Vec<u8>, Error> {
    let mut zin = ZipArchive::new(Cursor::new(bytes))?;
    let mut out = ZipWriter::new(Cursor::new(Vec::new()));

    for i in 0..zin.len() {
        let file = zin.by_index(i)?;
        let name = file.name().to_string();
        if let Some(profile) = profile_for(fmt, &name) {
            let mut data = Vec::with_capacity(usize::try_from(file.size()).unwrap_or(0));
            let mut file = file;
            file.read_to_end(&mut data)?;
            let new = process_part(&data, &profile, f)?;
            out.start_file(
                name,
                SimpleFileOptions::default()
                    .compression_method(zip::CompressionMethod::Deflated),
            )?;
            out.write_all(&new)?;
        } else {
            // Raw copy keeps compression and ordering (ODF's stored `mimetype`
            // entry must remain first and uncompressed).
            out.raw_copy_file(file)?;
        }
    }

    Ok(out.finish()?.into_inner())
}

/// Extract all detectable text (body, headers, notes, comments, metadata) as
/// newline-joined paragraphs — used by `nym detect` on office files.
pub fn extract_text(bytes: &[u8], fmt: OfficeFormat) -> Result<String, Error> {
    let mut paras: Vec<String> = Vec::new();
    rewrite_archive(bytes, fmt, &mut |para| {
        paras.push(para.to_string());
        None
    })?;
    Ok(paras.join("\n"))
}

/// Anonymize an office document in place. Returns the rewritten archive bytes
/// and the replacement log (for key files / reversibility).
pub fn anonymize(
    bytes: &[u8],
    fmt: OfficeFormat,
    detector: &Detector,
    replacer: &mut Replacer,
) -> Result<(Vec<u8>, Vec<Replacement>), Error> {
    let mut all: Vec<Replacement> = Vec::new();
    let out = rewrite_archive(bytes, fmt, &mut |para| {
        let matches = detector.detect(para);
        if matches.is_empty() {
            return None;
        }
        let mut spans: Vec<Span> = Vec::new();
        let mut last_end = 0usize;
        for m in &matches {
            if m.start < last_end {
                continue; // skip overlapping matches, same as Replacer::replace_all
            }
            let r = replacer.replace(m);
            spans.push((m.start, m.end, r.replacement.clone()));
            all.push(r);
            last_end = m.end;
        }
        Some(spans)
    })?;
    Ok((out, all))
}

/// Reverse a previous anonymization using `replacement -> original` pairs from
/// a key file. Returns the restored archive and the number of restorations.
pub fn deanonymize(
    bytes: &[u8],
    fmt: OfficeFormat,
    mappings: &[(String, String)],
) -> Result<(Vec<u8>, usize), Error> {
    // Longest replacement first so nested/overlapping candidates resolve
    // deterministically toward the most specific mapping.
    let mut sorted: Vec<&(String, String)> = mappings.iter().filter(|(k, _)| !k.is_empty()).collect();
    sorted.sort_by_key(|(k, _)| std::cmp::Reverse(k.len()));

    let mut count = 0usize;
    let out = rewrite_archive(bytes, fmt, &mut |para| {
        let mut spans: Vec<Span> = Vec::new();
        for (repl, original) in &sorted {
            let mut from = 0usize;
            while let Some(pos) = para[from..].find(repl.as_str()) {
                let s = from + pos;
                let e = s + repl.len();
                let overlaps = spans.iter().any(|&(a, b, _)| s < b && e > a);
                if !overlaps {
                    spans.push((s, e, original.clone()));
                }
                from = e;
            }
        }
        if spans.is_empty() {
            return None;
        }
        spans.sort_by_key(|&(s, _, _)| s);
        count += spans.len();
        Some(spans)
    })?;
    Ok((out, count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::replacer::ReplacerConfig;
    use crate::engine::{DetectorConfig, ReplacementStrategy};

    /// Build a minimal but structurally valid docx in memory.
    fn make_docx(document_xml: &str) -> Vec<u8> {
        let mut zw = ZipWriter::new(Cursor::new(Vec::new()));
        let opt = SimpleFileOptions::default();
        let content_types = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"#;
        let core = r#"<?xml version="1.0"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:creator>Hannah Meyer</dc:creator><cp:lastModifiedBy>Hannah Meyer</cp:lastModifiedBy>
</cp:coreProperties>"#;
        #[expect(clippy::unwrap_used, reason = "test fixture construction")]
        {
            zw.start_file("[Content_Types].xml", opt).unwrap();
            zw.write_all(content_types.as_bytes()).unwrap();
            zw.start_file("word/document.xml", opt).unwrap();
            zw.write_all(document_xml.as_bytes()).unwrap();
            zw.start_file("docProps/core.xml", opt).unwrap();
            zw.write_all(core.as_bytes()).unwrap();
        }
        #[expect(clippy::unwrap_used, reason = "test fixture construction")]
        zw.finish().unwrap().into_inner()
    }

    fn read_part(bytes: &[u8], name: &str) -> String {
        #[expect(clippy::unwrap_used, reason = "test assertion")]
        {
            let mut za = ZipArchive::new(Cursor::new(bytes)).unwrap();
            let mut f = za.by_name(name).unwrap();
            let mut s = String::new();
            f.read_to_string(&mut s).unwrap();
            s
        }
    }

    const DOC: &str = r#"<?xml version="1.0"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:r><w:t>Contact me at jo</w:t></w:r><w:r><w:t>hn.doe@ex</w:t></w:r><w:r><w:t>ample.com today.</w:t></w:r></w:p>
<w:p><w:r><w:t>Server 192.168.1.77 is fine.</w:t></w:r></w:p>
</w:body></w:document>"#;

    #[test]
    fn docx_split_run_email_is_redacted() {
        let bytes = make_docx(DOC);
        let detector = Detector::new(&DetectorConfig::default());
        let mut replacer = Replacer::new(ReplacerConfig {
            strategy: ReplacementStrategy::Placeholder,
            ..ReplacerConfig::default()
        });
        #[expect(clippy::unwrap_used, reason = "test")]
        let (out, replacements) = anonymize(&bytes, OfficeFormat::Docx, &detector, &mut replacer).unwrap();

        let doc = read_part(&out, "word/document.xml");
        assert!(!doc.contains("john.doe"), "email fragments must be gone: {doc}");
        assert!(!doc.contains("example.com"), "email tail must be gone: {doc}");
        assert!(doc.contains("&lt;EMAIL&gt;") || doc.contains("<EMAIL>"), "placeholder present: {doc}");
        assert!(doc.contains("192.168.1.77") || replacements.iter().any(|r| r.original == "192.168.1.77"));
        // XML must stay well-formed and the run structure intact.
        assert!(doc.contains("</w:p>"));
        assert!(replacements.iter().any(|r| r.original == "john.doe@example.com"));
    }

    #[test]
    fn docx_roundtrip_deanonymize_restores_original() {
        let bytes = make_docx(DOC);
        let detector = Detector::new(&DetectorConfig::default());
        let mut replacer = Replacer::new(ReplacerConfig {
            strategy: ReplacementStrategy::Placeholder,
            ..ReplacerConfig::default()
        });
        #[expect(clippy::unwrap_used, reason = "test")]
        let (anon, replacements) = anonymize(&bytes, OfficeFormat::Docx, &detector, &mut replacer).unwrap();

        let map: Vec<(String, String)> = replacements
            .iter()
            .map(|r| (r.replacement.clone(), r.original.clone()))
            .collect();
        #[expect(clippy::unwrap_used, reason = "test")]
        let (restored, n) = deanonymize(&anon, OfficeFormat::Docx, &map).unwrap();
        #[expect(clippy::unwrap_used, reason = "test")]
        let anon_text = extract_text(&anon, OfficeFormat::Docx).unwrap();
        let raw = read_part(&anon, "word/document.xml");
        assert!(n >= 1, "no restorations; anon text: {anon_text:?}; map: {map:?}; raw: {raw}");
        let doc = read_part(&restored, "word/document.xml");
        // The full email is restored, though possibly consolidated into one run.
        assert!(doc.contains("john.doe@ex") || doc.contains("john.doe@example.com"), "{doc}");
    }

    #[test]
    fn docx_metadata_author_is_covered_by_extract() {
        let bytes = make_docx(DOC);
        #[expect(clippy::unwrap_used, reason = "test")]
        let text = extract_text(&bytes, OfficeFormat::Docx).unwrap();
        assert!(text.contains("Hannah Meyer"), "metadata text extracted: {text}");
        assert!(text.contains("Contact me at john.doe@example.com today."), "split runs joined: {text}");
    }

    #[test]
    fn untouched_parts_are_byte_identical() {
        let bytes = make_docx(DOC);
        let detector = Detector::new(&DetectorConfig::default());
        let mut replacer = Replacer::new(ReplacerConfig::default());
        #[expect(clippy::unwrap_used, reason = "test")]
        let (out, _) = anonymize(&bytes, OfficeFormat::Docx, &detector, &mut replacer).unwrap();
        assert_eq!(
            read_part(&bytes, "[Content_Types].xml"),
            read_part(&out, "[Content_Types].xml")
        );
    }

    #[test]
    fn xlsx_shared_strings_redacted_formulas_untouched() {
        let mut zw = ZipWriter::new(Cursor::new(Vec::new()));
        let opt = SimpleFileOptions::default();
        let shared = r#"<?xml version="1.0"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="2" uniqueCount="2">
<si><t>alice@corp.example</t></si>
<si><r><t>call 555-</t></r><r><t>123-4567 now</t></r></si>
</sst>"#;
        let sheet = r#"<?xml version="1.0"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1"><f>SUM(A1:A9)</f><v>42</v></c></row>
</sheetData></worksheet>"#;
        #[expect(clippy::unwrap_used, reason = "test fixture")]
        {
            zw.start_file("xl/sharedStrings.xml", opt).unwrap();
            zw.write_all(shared.as_bytes()).unwrap();
            zw.start_file("xl/worksheets/sheet1.xml", opt).unwrap();
            zw.write_all(sheet.as_bytes()).unwrap();
        }
        #[expect(clippy::unwrap_used, reason = "test fixture")]
        let bytes = zw.finish().unwrap().into_inner();

        let detector = Detector::new(&DetectorConfig::default());
        let mut replacer = Replacer::new(ReplacerConfig {
            strategy: ReplacementStrategy::Placeholder,
            ..ReplacerConfig::default()
        });
        #[expect(clippy::unwrap_used, reason = "test")]
        let (out, _) = anonymize(&bytes, OfficeFormat::Xlsx, &detector, &mut replacer).unwrap();
        let sst = read_part(&out, "xl/sharedStrings.xml");
        assert!(!sst.contains("alice@corp.example"), "{sst}");
        let ws = read_part(&out, "xl/worksheets/sheet1.xml");
        assert!(ws.contains("SUM(A1:A9)"), "formula untouched: {ws}");
        assert!(ws.contains("<v>42</v>"), "value untouched: {ws}");
    }
}
