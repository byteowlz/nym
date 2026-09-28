//! nym-ocr: OCR engine companion for nym's raster redaction.
//!
//! Runs PP-OCRv6 (via liteparse `OarOcrEngine`, backed by oar-ocr/ONNX) on one
//! image and prints nym's OCR JSON contract on stdout:
//!
//! ```json
//! {"engine":"nym-ocr/pp-ocr","width":1240,"height":1754,
//!  "words":[{"text":"John","conf":0.98,"x":10,"y":20,"w":52,"h":18}]}
//! ```
//!
//! Boxes are detection-region level (pixel-true DBNet polygons, reduced to
//! their axis-aligned bounds). Models are resolved from Hugging Face
//! (`~/byteowlz/nym/tools/nym-ocr/src/models.rs`, ported from ingestr's
//! ADR-0004 store): official PaddlePaddle `PP-OCRv6_{tiny,small,medium}_{det,rec}`
//! ONNX exports, SHA-256 pinned, cached in the standard Hugging Face cache, and
//! looked up as `$NYM_SHARED_HF_HOME/hub` → user's `$HF_HOME/hub` → download
//! (unless `HF_HUB_OFFLINE=1`). The bare-name ModelScope auto-download is
//! deliberately not used.
//!
//! Model tier is selected with `NYM_OCR_TIER` (`tiny` | `small` | `medium`;
//! default `small`). The engine is built once per process (the expensive part is
//! model loading) and reused across pages/files within a single invocation.

mod models;

use std::path::Path;
use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow, bail};
use liteparse::ocr::oar::OarOcrEngine;
use liteparse::ocr::{OcrEngine, OcrOptions, OcrResult};
use serde::Serialize;

use models::PpOcrTier;

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

/// Process-wide PP-OCRv6 engine, built once and shared. The model store is
/// keyed by tier so a process that changes tiers doesn't silently keep the
/// first one it loaded.
fn ppocr_engine(tier: PpOcrTier) -> Result<&'static OarOcrEngine> {
    static ENGINE: OnceLock<Result<OarOcrEngine, String>> = OnceLock::new();
    ENGINE
        .get_or_init(|| {
            log::info!("loading PP-OCRv6 {}", tier.as_str());
            let models = models::ensure_ppocr(tier).map_err(|e| format!("{e:#}"))?;
            OarOcrEngine::from_models(
                models.det.as_path(),
                models.rec.as_path(),
                models.dict.as_bytes(),
            )
            .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|e| anyhow!("initializing PP-OCRv6 engine: {e}"))
}

fn block_on<T>(fut: impl std::future::Future<Output = T>) -> T {
    futures::executor::block_on(fut)
}

/// Convert a liteparse `OcrResult` (bbox `[x1,y1,x2,y2]`) to nym's word box.
fn to_word(r: &OcrResult) -> Word {
    let [x1, y1, x2, y2] = r.bbox;
    Word {
        text: r.text.clone(),
        conf: r.confidence,
        x: x1.max(0.0) as u32,
        y: y1.max(0.0) as u32,
        w: (x2 - x1).max(0.0) as u32,
        h: (y2 - y1).max(0.0) as u32,
    }
}

fn main() -> Result<()> {
    env_logger::init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!(
            "usage: nym-ocr <image> [image ...]\n\n\
            OCR one or more images; the model is loaded once for the whole\n\
            invocation. With multiple images (or `--batch`) a JSON array of\n\
            results is emitted, one per image in argument order.\n\n\
            model tier: NYM_OCR_TIER=tiny|small|medium (default small)"
        );
        return Ok(());
    }

    let mut batch = false;
    let mut images: Vec<String> = Vec::new();
    for a in &args {
        if a == "--batch" {
            batch = true;
        } else {
            images.push(a.clone());
        }
    }
    if images.is_empty() {
        bail!("usage: nym-ocr <image> [image ...]");
    }

    let tier = PpOcrTier::parse(&env_or("NYM_OCR_TIER", "small"));
    let engine = ppocr_engine(tier)?;

    let mut outputs = Vec::with_capacity(images.len());
    for img_path in &images {
        let img = image::open(Path::new(img_path))
            .with_context(|| format!("failed to load image: {img_path}"))?
            .into_rgb8();
        let (width, height) = img.dimensions();

        let options = OcrOptions {
            language: "en".to_string(),
            dpi: 300.0,
        };
        let results = block_on(engine.recognize(img.as_raw(), width, height, &options))
            .map_err(|e| anyhow!("OCR prediction failed on {img_path}: {e}"))?;

        let words: Vec<Word> = results.iter().map(to_word).collect();
        outputs.push(Output {
            engine: format!("nym-ocr/pp-ocr/{}", tier.as_str()),
            width,
            height,
            words,
        });
    }

    if batch || outputs.len() > 1 {
        println!("{}", serde_json::to_string(&outputs)?);
    } else {
        println!("{}", serde_json::to_string(&outputs[0])?);
    }
    Ok(())
}