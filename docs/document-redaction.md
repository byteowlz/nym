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
