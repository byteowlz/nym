# Document redaction (docx / xlsx / pptx / ODF / PDF)

nym redacts office documents and PDFs **in place** — same file format out,
formatting preserved, PII gone. No conversion step: just point `detect` /
`anon` (and for office formats, `deanon`) at the file.

```bash
nym detect report.docx                     # extract + scan all text incl. metadata
nym anon report.docx -k keys.jsonl         # -> report.anon.docx (reversible)
nym deanon report.anon.docx -k keys.jsonl  # -> report.anon.restored.docx
nym anon contract.pdf                      # -> contract.anon.pdf (true redaction)
```

Output defaults to `<name>.anon.<ext>` next to the input (`-o` to override).
All detection machinery applies: regex patterns, rulesets, `--ner` backends,
replacement strategies (`placeholder`, `fake`, `hash`, …), key files.

## Office formats (docx, xlsx, pptx, odt/ods/odp)

These are ZIP archives of XML; nym rewrites only the *text nodes* of the
text-bearing parts, so styles, tables, images and layout survive byte-identical.

What gets scanned and redacted:

| Format | Covered surfaces |
|--------|-----------------|
| docx | body, headers/footers, footnotes/endnotes, comments, core metadata (author, lastModifiedBy, title, …) |
| pptx | slides, speaker notes, comments, core metadata |
| xlsx | shared strings, inline strings, cell comments, core metadata — **formulas and numeric cell values are never touched** |
| odt/ods/odp | content, styles, meta |

Details that matter:

- **Split runs.** Word fragments text mid-word across runs (an email may live in
  three `<w:t>` nodes). Detection runs on the joined paragraph text and the
  replacement is mapped back onto the runs — a bold email becomes a bold
  placeholder; formatting boundaries survive.
- **Reversibility.** Office redaction is fully reversible via key files, exactly
  like plain text.
- Untouched archive entries are copied through byte-identical.

## PDF — true redaction

PDF redaction *removes the text from the content streams* — it never draws a
box over it (the classic redaction failure that leaves text extractable).

How it works: content streams (and Form XObjects — letterheads, stamps) are
decoded to operators; show-text strings are mapped to Unicode via each font's
`ToUnicode` CMap (or a Latin-1 approximation for simple fonts); the detector
runs on the assembled page text; matched spans are **cut out of the string
bytes**. When the placeholder (`<EMAIL>`, a fake value, …) is encodable in the
same font it is spliced in; otherwise the text is simply removed. The `Info`
dictionary, annotations and form-field values are scrubbed with the same
detector, and the XMP metadata stream is dropped.

**Verification is built in and mandatory**: the produced PDF is re-parsed and
every redacted value is searched for in the extracted text, all string objects,
every decompressed stream, and the raw bytes. If anything survives, nym refuses
to write the output.

Honest limitations (reported, not hidden):

- **Raster images are not scanned** (no OCR) — a scanned letter keeps its
  pixels. nym prints a note when images are present.
- Fonts without a usable Unicode mapping make their text uninspectable. By
  default nym **refuses to redact** such files; `--no-strict-pdf` proceeds and
  reports the number of skipped operators.
- **Encrypted PDFs are rejected** — decrypt first.
- **Redaction is destructive by design**: `deanon` is not available for PDFs.
  The key file documents what was removed; keep the original if you need it.
- Layout: removed text is not reflowed. With placeholders inserted the line
  re-kerns slightly; with pure removal a gap simply closes.

Verified against PDFs from fpdf2 (core fonts and embedded TTF subsets with
ToUnicode CMaps) and reportlab (compressed streams), with independent
extraction via pypdf confirming removal.

## OCR: images and scanned PDFs

nym also redacts **raster PII** — standalone images and scanned PDF pages —
using an **external OCR engine** (pluggable; engine churn stays out of nym)
while the safety-critical work stays inside nym: mapping matches to pixel
regions, painting, re-encoding, and verification.

```bash
nym detect scan.png                  # OCR the image, scan the recognized text
nym anon scan.png                    # paint PII regions -> scan.anon.png
nym anon scanned-contract.pdf --ocr  # also OCR+redact images inside the PDF
```

**Engines** (`[ocr] engine` in config, default `auto`):

- **`nym-ocr`** (recommended) — PP-OCR companion binary in this repo:
  `cargo install --path tools/nym-ocr`. Ships as a separate process because it
  uses a newer ONNX Runtime than nym's GLiNER backend allows; models
  auto-download to `~/.oar` on first use. Detection boxes are pixel-true
  DBNet regions.
- **`tesseract`** — used automatically if installed and nym-ocr is not.
- **Custom** — any command template with `{input}` that prints nym's OCR JSON
  contract: `{"words":[{"text":..,"conf":..,"x":..,"y":..,"w":..,"h":..}]}`.

**How redaction works**: recognize → detect PII on the assembled text → paint
black boxes over the matched regions (proportional sub-boxes with margin) →
**re-OCR the painted image**; if any redacted value is still recognized,
painting escalates to the full text regions and re-verifies — and if it still
survives, nym refuses to write the output. Like PDF text redaction this is
destructive: `deanon` refuses images, keep the original.

Notes:

- OCR noise is expected (`DOB` read as `D0B`) — pair with the `tokens` NER
  backend, whose training data includes OCR-style corruption, to catch
  mangled PII that exact regex misses.
- Scanned-PDF support currently covers **JPEG (DCTDecode) images** — what
  consumer scanners produce. CCITT/JBIG2 fax codecs are counted, reported,
  and refused in strict mode.
- Only the *values* found by detection are painted; surrounding text (form
  labels, headings) stays readable.

Verified end-to-end: PP-OCR recognition → painting → pikepdf-extracted image
from the redacted PDF re-OCR'd independently with no PII recognizable.
