//! Synthetic, offline contracts for verified decoded-XML restoration.

use super::*;
use crate::engine::replacer::ComponentMapping;

fn key(original: &str, alias: &str) -> Replacement {
    Replacement {
        original: original.into(),
        replacement: alias.into(),
        pattern_name: "person".into(),
        components: vec![
            ComponentMapping {
                original: "Original".into(),
                replacement: "Donald".into(),
                component_type: "first_name".into(),
            },
            ComponentMapping {
                original: "Person".into(),
                replacement: "Duck".into(),
                component_type: "last_name".into(),
            },
        ],
    }
}

fn archive(parts: &[(&str, &str)]) -> Vec<u8> {
    let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
    for (name, data) in parts {
        writer
            .start_file(*name, SimpleFileOptions::default())
            .unwrap();
        writer.write_all(data.as_bytes()).unwrap();
    }
    writer.finish().unwrap().into_inner()
}

fn part(bytes: &[u8], name: &str) -> String {
    let mut archive = ZipArchive::new(Cursor::new(bytes)).unwrap();
    let mut value = String::new();
    archive
        .by_name(name)
        .unwrap()
        .read_to_string(&mut value)
        .unwrap();
    value
}

fn verify(
    bytes: &[u8],
    format: OfficeFormat,
    entries: &[Replacement],
) -> Result<(Vec<u8>, usize), Error> {
    let verifier = RestoreVerifier::new(entries).unwrap();
    let mappings = entries
        .iter()
        .map(|entry| (entry.replacement.clone(), entry.original.clone()))
        .collect::<Vec<_>>();
    deanonymize_verified(bytes, format, &mappings, &verifier)
}

#[test]
fn office_verified_docx_split_runs_and_metadata_restore_without_cascade() {
    let entries = vec![key("Original DONALD DUCK", "Donald Duck")];
    let xml = "<w:document xmlns:w=\"urn:test\"><w:body><w:p><w:r><w:t>Donald </w:t></w:r><w:r><w:t>Duck</w:t></w:r></w:p></w:body></w:document>";
    let metadata = "<root><creator>Donald Duck</creator></root>";
    let untouched = "<types><entry name=\"unchanged\"/></types>";
    let bytes = archive(&[
        ("word/document.xml", xml),
        ("docProps/core.xml", metadata),
        ("[Content_Types].xml", untouched),
        ("word/media/image.dat", "untouched binary placeholder"),
    ]);
    let (out, count) = verify(&bytes, OfficeFormat::Docx, &entries).unwrap();
    assert_eq!((count, part(&out, "word/document.xml"), part(&out, "docProps/core.xml"), part(&out, "[Content_Types].xml"), part(&out, "word/media/image.dat")), (2, "<w:document xmlns:w=\"urn:test\"><w:body><w:p><w:r><w:t>Original DONALD DUCK</w:t></w:r><w:r><w:t></w:t></w:r></w:p></w:body></w:document>".into(), "<root><creator>Original DONALD DUCK</creator></root>".into(), untouched.into(), "untouched binary placeholder".into()));
}

#[test]
fn office_verified_case_partial_and_path_leftovers_fail_without_values() {
    let entries = vec![key("Original Person", "Donald Duck")];
    for text in [
        "DONALD DUCK",
        "Donald approved",
        "/home/duck/report",
        "Donald_Duck/report",
    ] {
        let xml = format!("<root><p><t>{text}</t></p></root>");
        let bytes = archive(&[("word/document.xml", &xml)]);
        let before = bytes.clone();
        let error = verify(&bytes, OfficeFormat::Docx, &entries)
            .unwrap_err()
            .to_string();
        assert!(error.contains("restore verification failed"));
        assert!(!error.contains(text));
        assert!(!error.contains("Original Person"));
        assert_eq!(bytes, before);
    }
}

#[test]
fn office_verified_cdata_and_numeric_entities_use_decoded_original_text() {
    let entries = vec![key("Original <Person> & DONALD DUCK", "Donald Duck")];
    let xml = "<root><p><t>Don&#97;ld </t><t><![CDATA[Duck]]></t></p><p><t>D&#79;NALD DUCK</t></p></root>";
    let bytes = archive(&[("word/document.xml", xml)]);
    assert!(verify(&bytes, OfficeFormat::Docx, &entries).is_err());
    let xml = "<root><p><t>Don&#97;ld </t><t><![CDATA[Duck]]></t></p></root>";
    let bytes = archive(&[("word/document.xml", xml)]);
    let (out, count) = verify(&bytes, OfficeFormat::Docx, &entries).unwrap();
    assert_eq!(
        (count, extract_text(&out, OfficeFormat::Docx).unwrap()),
        (1, "Original <Person> & DONALD DUCK".into())
    );
    assert_eq!(
        part(&out, "word/document.xml"),
        "<root><p><t>Original &lt;Person&gt; &amp; DONALD DUCK</t><t></t></p></root>"
    );
}

#[test]
fn office_verified_untouched_cdata_is_checked_not_bypassed() {
    let entries = vec![key("Original Person", "Donald Duck")];
    for xml in [
        "<root><p><t><![CDATA[DONALD DUCK]]></t></p></root>",
        "<root><formula><![CDATA[Donald Duck]]></formula></root>",
    ] {
        let bytes = archive(&[("word/document.xml", xml)]);
        assert!(verify(&bytes, OfficeFormat::Docx, &entries).is_err());
    }
}

#[test]
fn office_verified_xlsx_shared_and_inline_strings_restore_formulas_preserved() {
    let entries = vec![key("Original Person", "Donald Duck")];
    let shared = "<sst><si><r><t>Donald </t></r><r><t>Duck</t></r></si></sst>";
    let worksheet = "<worksheet><row><c><is><t>Donald Duck</t></is></c><c><f>SUM(A1:A9)</f><v>42</v></c></row></worksheet>";
    let comments = "<comments><comment><text><t>Donald Duck</t></text></comment></comments>";
    let bytes = archive(&[
        ("xl/sharedStrings.xml", shared),
        ("xl/worksheets/sheet1.xml", worksheet),
        ("xl/comments1.xml", comments),
    ]);
    let (out, count) = verify(&bytes, OfficeFormat::Xlsx, &entries).unwrap();
    assert_eq!((count, part(&out, "xl/sharedStrings.xml"), part(&out, "xl/worksheets/sheet1.xml"), part(&out, "xl/comments1.xml")), (3, "<sst><si><r><t>Original Person</t></r><r><t></t></r></si></sst>".into(), "<worksheet><row><c><is><t>Original Person</t></is></c><c><f>SUM(A1:A9)</f><v>42</v></c></row></worksheet>".into(), "<comments><comment><text><t>Original Person</t></text></comment></comments>".into()));
}

#[test]
fn office_verified_pptx_slides_notes_and_odf_mixed_content_are_supported() {
    let entries = vec![key("Original Person", "Donald Duck")];
    let slide = "<slide><p><r><t>Donald </t></r><r><t>Duck</t></r></p></slide>";
    let note = "<notes><p><t>Donald Duck</t></p></notes>";
    let bytes = archive(&[
        ("ppt/slides/slide1.xml", slide),
        ("ppt/notesSlides/notesSlide1.xml", note),
    ]);
    let (out, count) = verify(&bytes, OfficeFormat::Pptx, &entries).unwrap();
    assert_eq!(
        (
            count,
            part(&out, "ppt/slides/slide1.xml"),
            part(&out, "ppt/notesSlides/notesSlide1.xml")
        ),
        (
            2,
            "<slide><p><r><t>Original Person</t></r><r><t></t></r></p></slide>".into(),
            "<notes><p><t>Original Person</t></p></notes>".into()
        )
    );
    let content = "<document><p>Donald <span>Duck</span></p><h>Donald Duck</h></document>";
    let bytes = archive(&[
        ("mimetype", "application/vnd.oasis.opendocument.text"),
        ("content.xml", content),
        ("meta.xml", "<meta><creator>Donald Duck</creator></meta>"),
    ]);
    let (out, count) = verify(&bytes, OfficeFormat::Odf, &entries).unwrap();
    assert_eq!(
        (
            count,
            part(&out, "content.xml"),
            part(&out, "meta.xml"),
            part(&out, "mimetype")
        ),
        (
            3,
            "<document><p>Original Person<span></span></p><h>Original Person</h></document>".into(),
            "<meta><creator>Original Person</creator></meta>".into(),
            "application/vnd.oasis.opendocument.text".into()
        )
    );
    let mut zip = ZipArchive::new(Cursor::new(out)).unwrap();
    assert_eq!(zip.by_index(0).unwrap().name(), "mimetype");
}

#[test]
fn office_verified_unchanged_xml_attributes_names_relationships_formulas_and_paths_fail() {
    let entries = vec![key("Original Person", "Donald Duck")];
    for (format, name, xml) in [
        (
            OfficeFormat::Docx,
            "word/document.xml",
            "<root><p author=\"Donald Duck\"><t>safe</t></p></root>",
        ),
        (
            OfficeFormat::Docx,
            "word/document.xml",
            "<root><p><t author=\"D&#111;nald Duck\">safe</t></p></root>",
        ),
        (
            OfficeFormat::Docx,
            "word/document.xml",
            "<root><Donald>safe</Donald></root>",
        ),
        (
            OfficeFormat::Docx,
            "word/document.xml",
            "<root><p Donald=\"safe\"><t>safe</t></p></root>",
        ),
        (
            OfficeFormat::Docx,
            "word/_rels/document.xml.rels",
            "<Relationships><Relationship Target=\"/users/Donald/report\"/></Relationships>",
        ),
        (
            OfficeFormat::Docx,
            "word/custom.xml",
            "<root><custom>Donald Duck</custom></root>",
        ),
        (
            OfficeFormat::Docx,
            "word/custom.xml",
            "<root><custom>Don</custom><custom>ald Duck</custom></root>",
        ),
        (
            OfficeFormat::Xlsx,
            "xl/worksheets/sheet1.xml",
            "<worksheet><c><f>LOOKUP(&quot;Donald Duck&quot;)</f></c></worksheet>",
        ),
        (
            OfficeFormat::Xlsx,
            "xl/worksheets/sheet1.xml",
            "<worksheet><c><v>Donald Duck</v></c></worksheet>",
        ),
        (
            OfficeFormat::Odf,
            "content.xml",
            "<root><p>safe</p><!--Donald Duck--></root>",
        ),
        (
            OfficeFormat::Odf,
            "content.xml",
            "<?trace Donald Duck?><root><p>safe</p></root>",
        ),
        (OfficeFormat::Docx, "word/Donald Duck.xml", "<root/>"),
    ] {
        let bytes = archive(&[(name, xml)]);
        assert!(
            verify(&bytes, format, &entries).is_err(),
            "unchanged surface must fail"
        );
    }
}

#[test]
fn office_verified_malformed_unresolved_and_dtd_xml_fail_generically() {
    let entries = vec![key("Original Person", "Donald Duck")];
    for xml in [
        "<root><p><t>Donald Duck</t></p>",
        "<root><p><t>Donald Duck</p></t></root>",
        "<root><p><t>&unknown;</t></p></root>",
        "<!DOCTYPE root [<!ENTITY custom 'Donald Duck'>]><root><p><t>&custom;</t></p></root>",
        "<root/><root/>",
        "before<root/>",
        "<root/>after",
        "",
    ] {
        let bytes = archive(&[("word/document.xml", xml)]);
        assert_eq!(
            verify(&bytes, OfficeFormat::Docx, &entries)
                .unwrap_err()
                .to_string(),
            "invalid or unsupported XML in verified office restoration"
        );
    }
    assert_eq!(
        verify(b"not a zip", OfficeFormat::Docx, &entries)
            .unwrap_err()
            .to_string(),
        "invalid or unsupported XML in verified office restoration"
    );
}

#[test]
fn office_verified_no_supported_xml_part_cannot_silently_succeed() {
    let entries = vec![key("Original Person", "Donald Duck")];
    let bytes = archive(&[
        ("word/media/image.dat", "opaque fixture"),
        ("custom.xml", "<root/>"),
    ]);
    assert_eq!(
        verify(&bytes, OfficeFormat::Docx, &entries)
            .unwrap_err()
            .to_string(),
        "invalid or unsupported XML in verified office restoration"
    );
}

#[test]
fn office_verified_svg_vml_rdf_unchanged_xml_aliases_fail() {
    let entries = vec![key("Original Person", "Donald Duck")];
    for name in [
        "word/media/image.svg",
        "word/drawings/drawing.vml",
        "manifest.rdf",
    ] {
        let bytes = archive(&[
            ("word/document.xml", "<root><p><t>safe</t></p></root>"),
            (name, "<root><label>D&#79;NALD DUCK</label></root>"),
        ]);
        let error = verify(&bytes, OfficeFormat::Docx, &entries)
            .unwrap_err()
            .to_string();
        assert!(error.contains("restore verification failed"));
        assert!(!error.contains("DONALD DUCK"));
    }
}

#[test]
fn office_verified_short_alias_boundaries_and_original_alias_literals() {
    let entries = vec![key("Original Al", "Al")];
    let bytes = archive(&[(
        "word/document.xml",
        "<root><p><t>PAL Algae Al</t></p></root>",
    )]);
    let (out, count) = verify(&bytes, OfficeFormat::Docx, &entries).unwrap();
    assert_eq!(
        (count, part(&out, "word/document.xml")),
        (1, "<root><p><t>PAL Algae Original Al</t></p></root>".into())
    );
    let entries = vec![key("Original Unicode", "ΩΩ")];
    let bytes = archive(&[("word/document.xml", "<root><p><t>XΩΩ ΩΩ</t></p></root>")]);
    let (out, count) = verify(&bytes, OfficeFormat::Docx, &entries).unwrap();
    assert_eq!(
        (count, part(&out, "word/document.xml")),
        (1, "<root><p><t>XΩΩ Original Unicode</t></p></root>".into())
    );
}

#[test]
fn office_verified_non_cascade_and_component_mapping_rejection() {
    let entries = vec![key("alias-b", "alias-a"), key("actual-b", "alias-b")];
    let bytes = archive(&[(
        "word/document.xml",
        "<root><p><t>alias-a alias-b</t></p></root>",
    )]);
    let (out, count) = verify(&bytes, OfficeFormat::Docx, &entries).unwrap();
    assert_eq!(
        (count, part(&out, "word/document.xml")),
        (2, "<root><p><t>alias-b actual-b</t></p></root>".into())
    );
    let entries = vec![key("Original Person", "Donald Duck")];
    let verifier = RestoreVerifier::new(&entries).unwrap();
    let bytes = archive(&[("word/document.xml", "<root><p><t>Donald</t></p></root>")]);
    assert!(
        deanonymize_verified(
            &bytes,
            OfficeFormat::Docx,
            &[("Donald".into(), "Original".into())],
            &verifier
        )
        .is_err()
    );
}
