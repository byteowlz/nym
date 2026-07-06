//! nym-ocr: OCR engine companion for nym's raster redaction.
//!
//! Runs PP-OCR (via oar-ocr, models auto-downloaded to `$OAR_HOME`, default
//! `~/.oar`) on one image and prints nym's OCR JSON contract on stdout:
//!
//! ```json
//! {"engine":"nym-ocr/pp-ocr","width":1240,"height":1754,
//!  "words":[{"text":"John","conf":0.98,"x":10,"y":20,"w":52,"h":18}]}
//! ```
//!
//! Boxes are detection-region level (pixel-true DBNet polygons, reduced to
//! their axis-aligned bounds). Model files can be overridden with the
//! `NYM_OCR_DET` / `NYM_OCR_REC` / `NYM_OCR_DICT` environment variables
//! (bare names are auto-downloaded, paths are used as-is).

use anyhow::{Context, Result, bail};
use oar_ocr::prelude::*;
use serde::Serialize;
use std::path::Path;

#[derive(Serialize)]
struct Word {
    text: String,
    conf: f32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

#[derive(Serialize)]
struct Output {
    engine: String,
    width: u32,
    height: u32,
    words: Vec<Word>,
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!("usage: nym-ocr <image>\n(model overrides: NYM_OCR_DET / NYM_OCR_REC / NYM_OCR_DICT)");
        return Ok(());
    }
    let Some(img_path) = args.first() else {
        bail!("usage: nym-ocr <image>");
    };

    let det = env_or("NYM_OCR_DET", "pp-ocrv5_mobile_det.onnx");
    let rec = env_or("NYM_OCR_REC", "pp-ocrv5_mobile_rec.onnx");
    let dict = env_or("NYM_OCR_DICT", "ppocrv5_dict.txt");

    let ocr = OAROCRBuilder::new(det, rec, dict)
        .build()
        .context("failed to build OCR pipeline (models auto-download to $OAR_HOME, default ~/.oar)")?;

    let image = load_image(Path::new(img_path))
        .with_context(|| format!("failed to load image: {img_path}"))?;
    let (width, height) = (image.width(), image.height());

    let results = ocr.predict(vec![image]).context("OCR prediction failed")?;

    let mut words = Vec::new();
    for result in &results {
        for region in &result.text_regions {
            let (Some(text), Some(conf)) = (region.text.as_ref(), region.confidence) else {
                continue;
            };
            let pts = &region.bounding_box.points;
            if pts.is_empty() || text.trim().is_empty() {
                continue;
            }
            let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            for p in pts {
                x0 = x0.min(p.x);
                y0 = y0.min(p.y);
                x1 = x1.max(p.x);
                y1 = y1.max(p.y);
            }
            words.push(Word {
                text: text.to_string(),
                conf,
                x: x0.max(0.0) as u32,
                y: y0.max(0.0) as u32,
                w: (x1 - x0).max(0.0) as u32,
                h: (y1 - y0).max(0.0) as u32,
            });
        }
    }

    println!(
        "{}",
        serde_json::to_string(&Output {
            engine: "nym-ocr/pp-ocr".to_string(),
            width,
            height,
            words,
        })?
    );
    Ok(())
}
