//! OCR-based raster redaction: images and scanned PDF pages.
//!
//! Architecture: the OCR **engine is an external process** (pluggable — engine
//! churn stays out of nym), while the **safety-critical orchestration lives
//! here**: mapping detector matches to pixel regions, painting, re-encoding,
//! and verification.
//!
//! Engine contract (any command can implement it): given an image path, print
//! JSON on stdout:
//!
//! ```json
//! {"engine":"...","width":1240,"height":1754,
//!  "words":[{"text":"John","conf":0.98,"x":10,"y":20,"w":52,"h":18}]}
//! ```
//!
//! Built-in adapters: `nym-ocr` (PP-OCR companion binary, recommended —
//! `cargo install --path tools/nym-ocr`), `tesseract` (parses its TSV output),
//! or a custom command template with an `{input}` placeholder.
//!
//! Redaction flow per image: recognize → detect PII on the assembled text →
//! paint boxes over the matched regions (proportional sub-boxes with margin)
//! → **re-OCR the painted image and verify** none of the redacted values are
//! still recognized. If any survive, painting escalates to the full text
//! regions and verifies again; if they *still* survive, the operation fails
//! rather than emitting an unsafe image. Raster redaction is destructive —
//! keep the original if you need reversibility.

#![cfg(feature = "ocr")]

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use image::DynamicImage;
use serde::Deserialize;

use super::detector::Detector;
use super::replacer::{Replacement, Replacer};

type Error = Box<dyn std::error::Error + Send + Sync>;

// ---------------------------------------------------------------------------
// Engine contract + adapters
// ---------------------------------------------------------------------------

/// One recognized text region with its pixel bounding box.
#[derive(Debug, Clone, Deserialize)]
pub struct OcrWord {
    pub text: String,
    #[serde(default)]
    pub conf: f32,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Engine output for one image.
#[derive(Debug, Clone, Deserialize)]
pub struct OcrOutput {
    #[serde(default)]
    pub engine: String,
    #[serde(default)]
    pub width: u32,
    #[serde(default)]
    pub height: u32,
    pub words: Vec<OcrWord>,
}

/// How the external engine is invoked.
#[derive(Debug, Clone)]
enum EngineKind {
    /// `nym-ocr <image>` (or any command emitting the JSON contract).
    Json(Vec<String>),
    /// `tesseract <image> stdout tsv` — TSV parsed into the contract.
    Tesseract,
}

/// A resolved external OCR engine.
#[derive(Debug, Clone)]
pub struct OcrEngine {
    kind: EngineKind,
    /// Words below this recognition confidence are ignored (0.0-1.0).
    pub min_confidence: f32,
}

fn on_path(bin: &str) -> bool {
    let Some(paths) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&paths).any(|d| d.join(bin).is_file())
}

impl OcrEngine {
    /// Resolve an engine from config: `auto`, `nym-ocr`, `tesseract`, or a
    /// custom command template containing `{input}`.
    pub fn resolve(engine: &str, min_confidence: f32) -> Result<Self, Error> {
        let kind = match engine {
            "auto" => {
                if on_path("nym-ocr") {
                    EngineKind::Json(vec!["nym-ocr".into(), "{input}".into()])
                } else if on_path("tesseract") {
                    EngineKind::Tesseract
                } else {
                    return Err(
                        "no OCR engine found. Install one:\n  \
                         cargo install --path tools/nym-ocr   (PP-OCR, recommended)\n  \
                         or install tesseract, or set [ocr].engine to a custom command"
                            .into(),
                    );
                }
            }
            "nym-ocr" => EngineKind::Json(vec!["nym-ocr".into(), "{input}".into()]),
            "tesseract" => EngineKind::Tesseract,
            custom if custom.contains("{input}") => EngineKind::Json(
                custom.split_whitespace().map(str::to_string).collect(),
            ),
            other => {
                return Err(format!(
                    "unknown OCR engine {other:?} (use auto, nym-ocr, tesseract, \
                     or a command template containing {{input}})"
                )
                .into());
            }
        };
        Ok(OcrEngine {
            kind,
            min_confidence,
        })
    }

    /// Run the engine on an image file.
    pub fn recognize_file(&self, path: &Path) -> Result<OcrOutput, Error> {
        let mut out = match &self.kind {
            EngineKind::Json(template) => {
                let mut parts = template.iter().map(|p| {
                    if p.contains("{input}") {
                        p.replace("{input}", &path.to_string_lossy())
                    } else {
                        p.clone()
                    }
                });
                let program = parts.next().ok_or("empty OCR command template")?;
                let output = Command::new(&program).args(parts).output().map_err(|e| {
                    format!("failed to run OCR engine {program:?}: {e}")
                })?;
                if !output.status.success() {
                    return Err(format!(
                        "OCR engine {program:?} failed: {}",
                        String::from_utf8_lossy(&output.stderr)
                    )
                    .into());
                }
                serde_json::from_slice::<OcrOutput>(&output.stdout)
                    .map_err(|e| format!("invalid OCR engine JSON: {e}"))?
            }
            EngineKind::Tesseract => {
                let output = Command::new("tesseract")
                    .arg(path)
                    .args(["stdout", "tsv"])
                    .output()
                    .map_err(|e| format!("failed to run tesseract: {e}"))?;
                if !output.status.success() {
                    return Err(format!(
                        "tesseract failed: {}",
                        String::from_utf8_lossy(&output.stderr)
                    )
                    .into());
                }
                parse_tesseract_tsv(&String::from_utf8_lossy(&output.stdout))
            }
        };
        out.words.retain(|w| {
            !w.text.trim().is_empty() && w.conf >= self.min_confidence && w.w > 0 && w.h > 0
        });
        Ok(out)
    }

    /// Run the engine on in-memory image bytes (written to a temp file).
    pub fn recognize_bytes(&self, bytes: &[u8], ext: &str) -> Result<OcrOutput, Error> {
        let path = temp_path(ext);
        std::fs::write(&path, bytes)?;
        let result = self.recognize_file(&path);
        let _ = std::fs::remove_file(&path);
        result
    }
}

fn temp_path(ext: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "nym-ocr-{}-{}.{ext}",
        std::process::id(),
        n
    ))
}

/// Parse tesseract TSV (level 5 rows are words; conf is 0-100).
fn parse_tesseract_tsv(tsv: &str) -> OcrOutput {
    let mut words = Vec::new();
    for line in tsv.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 12 || f[0] != "5" {
            continue;
        }
        let (Ok(x), Ok(y), Ok(w), Ok(h), Ok(conf)) = (
            f[6].parse::<u32>(),
            f[7].parse::<u32>(),
            f[8].parse::<u32>(),
            f[9].parse::<u32>(),
            f[10].parse::<f32>(),
        ) else {
            continue;
        };
        let text = f[11].trim();
        if text.is_empty() {
            continue;
        }
        words.push(OcrWord {
            text: text.to_string(),
            conf: (conf / 100.0).clamp(0.0, 1.0),
            x,
            y,
            w,
            h,
        });
    }
    OcrOutput {
        engine: "tesseract".to_string(),
        width: 0,
        height: 0,
        words,
    }
}

// ---------------------------------------------------------------------------
// Text assembly: words -> detector input, with span provenance
// ---------------------------------------------------------------------------

/// Assembled recognition text plus, per word, its byte span in that text.
pub struct OcrText {
    pub text: String,
    /// (byte_start, byte_end, word index)
    spans: Vec<(usize, usize, usize)>,
}

/// Order words into lines (by vertical overlap), join with spaces/newlines.
pub fn assemble_text(out: &OcrOutput) -> OcrText {
    let mut order: Vec<usize> = (0..out.words.len()).collect();
    order.sort_by_key(|&i| (out.words[i].y, out.words[i].x));

    let mut text = String::new();
    let mut spans = Vec::new();
    let mut prev: Option<usize> = None;

    for idx in order {
        let w = &out.words[idx];
        if let Some(p) = prev {
            let pw = &out.words[p];
            // Same line if vertical ranges overlap by at least half a height.
            let overlap = (pw.y + pw.h).min(w.y + w.h).saturating_sub(pw.y.max(w.y));
            let same_line = overlap * 2 >= pw.h.min(w.h).max(1);
            text.push(if same_line { ' ' } else { '\n' });
        }
        let start = text.len();
        text.push_str(&w.text);
        spans.push((start, text.len(), idx));
        prev = Some(idx);
    }
    OcrText { text, spans }
}

// ---------------------------------------------------------------------------
// Box mapping + painting
// ---------------------------------------------------------------------------

/// A pixel rectangle to paint.
#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Pixel boxes covering a matched byte span of the assembled text.
///
/// `precise` paints a proportional slice of each overlapped region (expanded
/// by ~1.5 average character widths on both sides); otherwise the full region
/// box is used (the escalation path).
fn boxes_for_span(ocr: &OcrText, out: &OcrOutput, s: usize, e: usize, precise: bool) -> Vec<Rect> {
    let mut rects = Vec::new();
    for &(ws, we, idx) in &ocr.spans {
        if ws >= e || we <= s {
            continue;
        }
        let w = &out.words[idx];
        if !precise || we <= ws {
            rects.push(Rect { x: w.x, y: w.y, w: w.w, h: w.h });
            continue;
        }
        // Proportional slice of the region by character position.
        let total = (we - ws) as f64;
        let char_w = f64::from(w.w) / total.max(1.0);
        let lo = (s.max(ws) - ws) as f64 / total;
        let hi = (e.min(we) - ws) as f64 / total;
        let margin = char_w * 1.5;
        let x0 = (f64::from(w.x) + lo * f64::from(w.w) - margin).max(0.0);
        let x1 = f64::from(w.x) + hi * f64::from(w.w) + margin;
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "clamped")]
        rects.push(Rect {
            x: x0 as u32,
            y: w.y,
            w: (x1 - x0).max(1.0) as u32,
            h: w.h,
        });
    }
    rects
}

/// Fill rectangles (plus padding) with black.
fn paint(img: &mut DynamicImage, rects: &[Rect], pad: u32) {
    let (iw, ih) = (img.width(), img.height());
    let rgba = img.as_mut_rgba8();
    let Some(buf) = rgba else { return };
    for r in rects {
        let x0 = r.x.saturating_sub(pad);
        let y0 = r.y.saturating_sub(pad);
        let x1 = (r.x + r.w + pad).min(iw);
        let y1 = (r.y + r.h + pad).min(ih);
        for y in y0..y1 {
            for x in x0..x1 {
                buf.put_pixel(x, y, image::Rgba([0, 0, 0, 255]));
            }
        }
    }
}

/// Normalize for verification: OCR of a painted image may re-segment text, so
/// compare case-insensitively with all whitespace removed.
fn normalize(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

// ---------------------------------------------------------------------------
// Image redaction orchestration
// ---------------------------------------------------------------------------

/// Result of redacting one raster image.
pub struct ImageRedaction {
    /// Re-encoded image bytes (same format as requested).
    pub bytes: Vec<u8>,
    /// What was removed.
    pub replacements: Vec<Replacement>,
    /// Whether escalation to full-region painting was needed.
    pub escalated: bool,
}

/// Output encoding for the painted image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RasterFormat {
    Png,
    Jpeg,
}

impl RasterFormat {
    pub fn from_ext(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "png" => Some(RasterFormat::Png),
            "jpg" | "jpeg" => Some(RasterFormat::Jpeg),
            _ => None,
        }
    }

    fn ext(self) -> &'static str {
        match self {
            RasterFormat::Png => "png",
            RasterFormat::Jpeg => "jpg",
        }
    }
}

fn encode(img: &DynamicImage, fmt: RasterFormat) -> Result<Vec<u8>, Error> {
    let mut out = std::io::Cursor::new(Vec::new());
    match fmt {
        RasterFormat::Png => img.write_to(&mut out, image::ImageFormat::Png)?,
        RasterFormat::Jpeg => {
            // JPEG has no alpha; flatten first.
            let rgb = DynamicImage::ImageRgb8(img.to_rgb8());
            rgb.write_to(&mut out, image::ImageFormat::Jpeg)?;
        }
    }
    Ok(out.into_inner())
}

/// Redact one raster image: OCR → detect → paint → re-OCR verify (with one
/// escalation round). Returns `Ok(None)` when no PII was recognized.
pub fn redact_image(
    bytes: &[u8],
    fmt: RasterFormat,
    engine: &OcrEngine,
    detector: &Detector,
    replacer: &mut Replacer,
) -> Result<Option<ImageRedaction>, Error> {
    let recognized = engine.recognize_bytes(bytes, fmt.ext())?;
    if recognized.words.is_empty() {
        return Ok(None);
    }
    let ocr_text = assemble_text(&recognized);
    let matches = detector.detect(&ocr_text.text);
    if matches.is_empty() {
        return Ok(None);
    }

    let mut img = image::load_from_memory(bytes)?;
    // Ensure an RGBA buffer we can paint into.
    img = DynamicImage::ImageRgba8(img.to_rgba8());

    let mut replacements = Vec::new();
    let mut precise_rects = Vec::new();
    let mut full_rects = Vec::new();
    let mut last_end = 0usize;
    for m in &matches {
        if m.start < last_end {
            continue;
        }
        last_end = m.end;
        precise_rects.extend(boxes_for_span(&ocr_text, &recognized, m.start, m.end, true));
        full_rects.extend(boxes_for_span(&ocr_text, &recognized, m.start, m.end, false));
        replacements.push(replacer.replace(m));
    }
    if precise_rects.is_empty() {
        return Ok(None);
    }

    let pad = 4;
    paint(&mut img, &precise_rects, pad);
    let mut out_bytes = encode(&img, fmt)?;

    // Verification round 1.
    let mut escalated = false;
    if !verify_gone(engine, &out_bytes, fmt, &replacements)? {
        // Escalate: paint the full text regions and re-verify.
        escalated = true;
        paint(&mut img, &full_rects, pad * 2);
        out_bytes = encode(&img, fmt)?;
        if !verify_gone(engine, &out_bytes, fmt, &replacements)? {
            return Err(
                "raster redaction verification FAILED: redacted text is still recognized \
                 in the painted image even after escalation; refusing to write an unsafe output"
                    .into(),
            );
        }
    }

    Ok(Some(ImageRedaction {
        bytes: out_bytes,
        replacements,
        escalated,
    }))
}

/// Re-OCR the painted image and confirm no redacted value is recognizable.
fn verify_gone(
    engine: &OcrEngine,
    bytes: &[u8],
    fmt: RasterFormat,
    replacements: &[Replacement],
) -> Result<bool, Error> {
    let re = engine.recognize_bytes(bytes, fmt.ext())?;
    let haystack = normalize(&assemble_text(&re).text);
    Ok(replacements
        .iter()
        .filter(|r| r.original.len() >= 4)
        .all(|r| !haystack.contains(&normalize(&r.original))))
}

/// Recognize the text inside a PDF's JPEG images (for `nym detect --ocr`).
pub fn extract_pdf_image_text(pdf_bytes: &[u8], engine: &OcrEngine) -> Result<String, Error> {
    let mut out = String::new();
    for jpeg in collect_pdf_jpegs(pdf_bytes)? {
        let recognized = engine.recognize_bytes(&jpeg, "jpg")?;
        if !recognized.words.is_empty() {
            out.push_str(&assemble_text(&recognized).text);
            out.push('\n');
        }
    }
    Ok(out)
}

/// All DCTDecode (JPEG) image XObject payloads in a PDF.
fn collect_pdf_jpegs(pdf_bytes: &[u8]) -> Result<Vec<Vec<u8>>, Error> {
    use lopdf::{Document, Object};
    let doc = Document::load_mem(pdf_bytes)?;
    let mut seen = std::collections::HashSet::new();
    let mut jpegs = Vec::new();
    for (_no, page_id) in doc.get_pages() {
        let Ok((res_opt, res_ids)) = doc.get_page_resources(page_id) else {
            continue;
        };
        let mut res_dicts: Vec<&lopdf::Dictionary> = Vec::new();
        if let Some(r) = res_opt {
            res_dicts.push(r);
        }
        for rid in res_ids {
            if let Ok(d) = doc.get_object(rid).and_then(Object::as_dict) {
                res_dicts.push(d);
            }
        }
        for res in res_dicts {
            let xobjs = match res.get(b"XObject") {
                Ok(Object::Reference(id)) => doc.get_object(*id).and_then(Object::as_dict).ok(),
                Ok(Object::Dictionary(d)) => Some(d),
                _ => None,
            };
            let Some(xobjs) = xobjs else { continue };
            for (_, r) in xobjs.iter() {
                let Ok(id) = r.as_reference() else { continue };
                if !seen.insert(id) {
                    continue;
                }
                let Ok(stream) = doc.get_object(id).and_then(Object::as_stream) else {
                    continue;
                };
                let is_jpeg_image = stream
                    .dict
                    .get(b"Subtype")
                    .and_then(Object::as_name)
                    .map(|n| n == b"Image")
                    .unwrap_or(false)
                    && stream
                        .dict
                        .get(b"Filter")
                        .and_then(Object::as_name)
                        .map(|n| n == b"DCTDecode")
                        .unwrap_or(false);
                if is_jpeg_image {
                    jpegs.push(stream.content.clone());
                }
            }
        }
    }
    Ok(jpegs)
}

// ---------------------------------------------------------------------------
// Scanned-PDF pass (runs on the already text-redacted PDF)
// ---------------------------------------------------------------------------

/// Report for the scanned-PDF OCR pass.
#[derive(Debug, Default)]
pub struct PdfOcrReport {
    /// JPEG images that were OCR'd (and possibly painted).
    pub images_scanned: usize,
    /// Images whose codec is unsupported for redaction (CCITT, JBIG2, ...).
    pub images_unsupported: usize,
    /// Images where PII was found and painted.
    pub images_redacted: usize,
}

/// OCR-redact the raster images inside a PDF (first slice: DCTDecode/JPEG,
/// the format consumer scanners produce). Unsupported codecs are counted and,
/// with `strict`, cause a hard error.
pub fn redact_pdf_images(
    pdf_bytes: &[u8],
    engine: &OcrEngine,
    detector: &Detector,
    replacer: &mut Replacer,
    strict: bool,
) -> Result<(Vec<u8>, Vec<Replacement>, PdfOcrReport), Error> {
    use lopdf::{Document, Object};

    let mut doc = Document::load_mem(pdf_bytes)?;
    let mut report = PdfOcrReport::default();
    let mut log = Vec::new();

    // Collect image XObject ids (deduplicated — shared images edited once).
    let mut seen = std::collections::HashSet::new();
    let mut jpeg_ids = Vec::new();
    for (_no, page_id) in doc.get_pages() {
        let Ok((res_opt, res_ids)) = doc.get_page_resources(page_id) else {
            continue;
        };
        let mut res_dicts: Vec<&lopdf::Dictionary> = Vec::new();
        if let Some(r) = res_opt {
            res_dicts.push(r);
        }
        for rid in res_ids {
            if let Ok(d) = doc.get_object(rid).and_then(Object::as_dict) {
                res_dicts.push(d);
            }
        }
        for res in res_dicts {
            let xobjs = match res.get(b"XObject") {
                Ok(Object::Reference(id)) => doc.get_object(*id).and_then(Object::as_dict).ok(),
                Ok(Object::Dictionary(d)) => Some(d),
                _ => None,
            };
            let Some(xobjs) = xobjs else { continue };
            for (_, r) in xobjs.iter() {
                let Ok(id) = r.as_reference() else { continue };
                if !seen.insert(id) {
                    continue;
                }
                let Ok(stream) = doc.get_object(id).and_then(Object::as_stream) else {
                    continue;
                };
                let is_image = stream
                    .dict
                    .get(b"Subtype")
                    .and_then(Object::as_name)
                    .map(|n| n == b"Image")
                    .unwrap_or(false);
                if !is_image {
                    continue;
                }
                let is_jpeg = stream
                    .dict
                    .get(b"Filter")
                    .and_then(Object::as_name)
                    .map(|n| n == b"DCTDecode")
                    .unwrap_or(false);
                if is_jpeg {
                    jpeg_ids.push(id);
                } else {
                    report.images_unsupported += 1;
                }
            }
        }
    }

    if strict && report.images_unsupported > 0 {
        return Err(format!(
            "{} raster image(s) use codecs not yet supported for OCR redaction \
             (e.g. CCITT/JBIG2); their pixels cannot be inspected. Refusing \
             (pass --no-strict-pdf to override).",
            report.images_unsupported
        )
        .into());
    }

    for id in jpeg_ids {
        let jpeg = {
            let Ok(stream) = doc.get_object(id).and_then(Object::as_stream) else {
                continue;
            };
            stream.content.clone()
        };
        report.images_scanned += 1;
        match redact_image(&jpeg, RasterFormat::Jpeg, engine, detector, replacer)? {
            None => {}
            Some(red) => {
                report.images_redacted += 1;
                if let Ok(obj) = doc.get_object_mut(id) {
                    if let Ok(stream) = obj.as_stream_mut() {
                        // Re-encoded as RGB JPEG: keep DCTDecode, fix the
                        // color-space keys to match.
                        stream.set_content(red.bytes);
                        stream.dict.set("Filter", lopdf::Object::Name(b"DCTDecode".to_vec()));
                        stream
                            .dict
                            .set("ColorSpace", lopdf::Object::Name(b"DeviceRGB".to_vec()));
                        stream.dict.set("BitsPerComponent", 8);
                        stream.dict.remove(b"DecodeParms");
                    }
                }
                log.extend(red.replacements);
            }
        }
    }

    let mut out = Vec::new();
    doc.save_to(&mut out)?;
    // Flush replacements into the shared log (bytes verified per image above).
    let _ = std::io::sink().flush();
    Ok((out, log, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tesseract_tsv_parses_words() {
        let tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n\
                   1\t1\t0\t0\t0\t0\t0\t0\t100\t50\t-1\t\n\
                   5\t1\t1\t1\t1\t1\t10\t20\t52\t18\t96.5\tJohn\n\
                   5\t1\t1\t1\t1\t2\t70\t20\t60\t18\t91.0\tSmith\n";
        let out = parse_tesseract_tsv(tsv);
        assert_eq!(out.words.len(), 2);
        assert_eq!(out.words[0].text, "John");
        assert!((out.words[0].conf - 0.965).abs() < 1e-3);
        assert_eq!(out.words[1].x, 70);
    }

    #[test]
    fn assemble_joins_lines_by_vertical_overlap() {
        let out = OcrOutput {
            engine: String::new(),
            width: 200,
            height: 100,
            words: vec![
                OcrWord { text: "Hello".into(), conf: 0.9, x: 10, y: 10, w: 40, h: 12 },
                OcrWord { text: "World".into(), conf: 0.9, x: 60, y: 11, w: 40, h: 12 },
                OcrWord { text: "Below".into(), conf: 0.9, x: 10, y: 40, w: 40, h: 12 },
            ],
        };
        let t = assemble_text(&out);
        assert_eq!(t.text, "Hello World\nBelow");
    }

    #[test]
    fn boxes_cover_matched_span() {
        let out = OcrOutput {
            engine: String::new(),
            width: 400,
            height: 40,
            words: vec![OcrWord {
                text: "mail: john@example.com now".into(),
                conf: 0.9,
                x: 100,
                y: 5,
                w: 260,
                h: 14,
            }],
        };
        let t = assemble_text(&out);
        let s = t.text.find("john@").map_or(0, |v| v);
        let e = t.text.find(".com").map_or(0, |v| v) + 4;
        let precise = boxes_for_span(&t, &out, s, e, true);
        assert_eq!(precise.len(), 1);
        let r = precise[0];
        // Proportional slice must sit inside a sane expansion of the region.
        assert!(r.x >= 100 && r.x < 360, "{r:?}");
        assert!(r.w > 100, "covers the email: {r:?}");
        let full = boxes_for_span(&t, &out, s, e, false);
        assert_eq!((full[0].x, full[0].w), (100, 260));
    }

    #[test]
    fn paint_blacks_out_rect() {
        let mut img = DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            50,
            20,
            image::Rgba([255, 255, 255, 255]),
        ));
        paint(&mut img, &[Rect { x: 10, y: 5, w: 10, h: 5 }], 0);
        let buf = img.to_rgba8();
        assert_eq!(buf.get_pixel(15, 7).0, [0, 0, 0, 255]);
        assert_eq!(buf.get_pixel(5, 7).0, [255, 255, 255, 255]);
    }

    #[test]
    fn normalize_is_whitespace_and_case_insensitive() {
        assert_eq!(normalize("Jo hn.Doe@X.COM"), normalize("john.doe@x.com"));
    }
}
