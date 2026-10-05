//! Decision-model adjudication layer.
//!
//! nym's regex + NER layers are deterministic and fast, but they have a shared
//! blind spot: *unlabeled* secrets and context-dependent "is this really
//! private?" calls that carry no name and no recognisable structure. This
//! module runs a System-One style **decision model** over candidate spans and
//! asks typed questions (`is_secret` / `class` / `over_redacted`) whose answers
//! are read back as a small JSON object.
//!
//! The decision model is deliberately a *gate/adjudicator*, not a reasoner: it
//! never generates long prose, it only classifies a span as `redact` / `keep` /
//! `flag`, with a calibrated confidence. It is meant to sit **after** the
//! deterministic layers as the residual catch for the unlabeled class, and to
//! **veto** deterministic over-redactions (e.g. the generic `api_key` regex
//! firing on JWT/base64 fragments).
//!
//! Endpoint: any OpenAI-compatible `/v1/chat/completions`. KEYS ARE NEVER
//! STORED IN THE REPO -- an optional API key is read from an environment
//! variable named by [`DecisionConfig`].
//!
//! Privacy: only the snippet around each candidate (bounded by
//! [`DecisionConfig::context_chars`]) leaves the machine, never the whole
//! document.

use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use super::{Confidence, PiiMatch};

/// Verdict a decision model returns for one candidate span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Real secret/credential/PII -- should be redacted.
    Redact,
    /// Benign (code sample, path, uuid, checksum) -- keep as-is.
    Keep,
    /// Ambiguous -- surface for manual review rather than deciding either way.
    Flag,
}

/// One adjudication decision for a single candidate span.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    /// Byte range in the input this decision covers, if it came from regex/NER.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end: Option<usize>,
    /// The candidate text that was adjudicated (truncated).
    pub text: String,
    /// The decision.
    pub verdict: Verdict,
    /// The class the model assigned (e.g. `aws_key`, `credential`, `path`,
    /// `code_sample`, `benign`).
    pub class: String,
    /// Model-confidence in `[0,1]`.
    pub confidence: f32,
    /// Optional human-readable rational (from the model), truncated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Where the candidate came from: `regex`, `ner`, or the high-entropy
    /// backstop (`entropy`).
    pub source: String,
    /// The pattern name that matched (regex/ner), if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pattern_name: Option<String>,
    /// The confidence the detector assigned, if from regex/NER.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detector_confidence: Option<Confidence>,
}

/// Configuration for the decision-model adjudication layer.
///
/// Loaded from `[decision]` in the config file; overridable on the CLI.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DecisionConfig {
    /// Enable the decision stage (requires the `decision` feature).
    pub enabled: bool,
    /// OpenAI-compatible chat-completions endpoint.
    pub endpoint: String,
    /// Model id to ask.
    pub model: String,
    /// Environment variable holding the API key (optional; read at runtime).
    pub api_key_env: Option<String>,
    /// Per-request timeout in seconds.
    pub timeout_secs: u64,
    /// p(secret) at or above which a candidate is adjudicated `redact`.
    pub threshold: f32,
    /// Characters of surrounding context sent with each candidate.
    pub context_chars: usize,
    /// Maximum number of candidates adjudicated in one run (0 = unlimited).
    pub max_candidates: usize,
    /// Enable the high-entropy unlabeled-secret backstop.
    pub entropy_backstop: bool,
    /// Batch size for a single HTTP request (how many candidates asked at once).
    pub batch_size: usize,
    /// Decision backend. `chat` = OpenAI-compatible `/v1/chat/completions`
    /// (label-only, decoded-token confidence, NOT calibrated).
    /// `systemone` = TypeSafe/Jev-compatible `/v1/systemone` (Choice/Noul/Score
    /// readout -> true per-option probabilities + derived confidence).
    /// Defaults to `chat`.
    pub backend: String,
}

impl Default for DecisionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: String::new(),
            model: "deepseek-v4-flash-vision".to_string(),
            api_key_env: None,
            timeout_secs: 30,
            threshold: 0.5,
            context_chars: 160,
            max_candidates: 0,
            entropy_backstop: true,
            batch_size: 1,
            backend: "chat".to_string(),
        }
    }
}

impl DecisionConfig {
    /// True when the stage is usable (enabled and has an endpoint).
    pub fn usable(&self) -> bool {
        self.enabled && !self.endpoint.is_empty()
    }
}

/// A generated candidate span for the unlabeled-secret backstop.
#[derive(Debug, Clone)]
struct Candidate {
    start: usize,
    end: usize,
    text: String,
    source: String,
    pattern_name: Option<String>,
    detector_confidence: Option<Confidence>,
}

/// Manual request shape sent to the decision endpoint (kept small/narrow so the
/// model answers with a short JSON object rather than prose).
#[derive(Serialize)]
struct DecisionRequest {
    model: String,
    messages: Vec<DecisionMessage>,
    temperature: f32,
    max_tokens: usize,
    response_format: ResponseFormat,
}

#[derive(Serialize)]
struct DecisionMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: String,
}

/// Response envelope from an OpenAI-compatible chat-completions endpoint.
#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: RawMessage,
}

#[derive(Deserialize)]
struct RawMessage {
    content: String,
}

/// The model's structured answer for one candidate.
#[derive(Deserialize)]
struct ModelAnswer {
    #[serde(default)]
    index: Option<usize>,
    verdict: String,
    class: String,
    #[serde(default)]
    confidence: f32,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    #[expect(
        dead_code,
        reason = "model may supply is_secret; kept for forward-compat"
    )]
    is_secret: Option<bool>,
}

/// Response envelope from a TypeSafe/Jev-compatible `/v1/systemone` endpoint.
#[derive(Deserialize)]
struct SystemOneResponse {
    answers: std::collections::HashMap<String, SystemOneAnswer>,
}

/// One typed answer from a `/v1/systemone` endpoint.
///
/// We send a `choice` per candidate (criteria `redact`/`keep`/`flag`), so the
/// answer carries `choice` + `probabilities` + `confidence`. Noul and Score
/// variants are supported structurally but are not used by the gate's choice
/// contract.
#[derive(Deserialize)]
struct SystemOneAnswer {
    #[serde(rename = "type")]
    qtype: String,
    #[serde(default)]
    choice: Option<String>,
    #[serde(default)]
    noul: Option<f32>,
    #[serde(default)]
    #[expect(
        dead_code,
        reason = "part of the /v1/systemone Score variant payload; not used by the gate's choice contract"
    )]
    score: Option<f32>,
    #[serde(default)]
    confidence: Option<f32>,
    #[serde(default)]
    probabilities: std::collections::HashMap<String, f32>,
}

impl SystemOneAnswer {
    /// Map a typed Jev answer onto a [`ModelAnswer`] so the existing
    /// [`apply_answer`] logic can consume it unchanged.
    ///
    /// For a `choice` we take `choice` as the verdict option
    /// (`redact`/`keep`/`flag`), the option probability as confidence, and the
    /// most-probable other option's class via the option key. For `noul` we
    /// threshold the 0-1 value against 0.5 to derive a redact/keep verdict. For
    /// `score` we treat the expected value as a confidence-like signal.
    fn as_decision(&self) -> Option<ModelAnswer> {
        match self.qtype.as_str() {
            "choice" => {
                // Verdict option = the chosen criteria key (redact/keep/flag).
                let verdict = self.choice.clone()?;
                let confidence = self.confidence.unwrap_or(0.0);
                Some(ModelAnswer {
                    index: None,
                    verdict: verdict.clone(),
                    // Class option = most-probable non-choice option; fall back
                    // to the choice option itself (already the verdict).
                    class: self
                        .probabilities
                        .iter()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map_or_else(|| verdict.clone(), |(k, _)| k.clone()),
                    confidence,
                    reason: None,
                    is_secret: None,
                })
            }
            "noul" => {
                // 0-1 value; threshold at 0.5 into a redact/keep verdict.
                let v = self.noul.unwrap_or(0.5);
                Some(ModelAnswer {
                    index: None,
                    verdict: if v >= 0.5 { "redact" } else { "keep" }.to_string(),
                    class: if v >= 0.5 { "secret" } else { "benign" }.to_string(),
                    confidence: v,
                    reason: None,
                    is_secret: None,
                })
            }
            _ => None,
        }
    }
}

/// The decision-model adjudicator.
///
/// Create with [`DecisionGate::new`], then call [`DecisionGate::adjudicate`]
/// over a document. It combines the deterministic matches (passed in) with an
/// optional high-entropy backstop, asks the model about each candidate, and
/// returns a [`Decision`] per candidate.
pub struct DecisionGate {
    config: DecisionConfig,
}

impl DecisionGate {
    /// Build a gate from the given configuration.
    pub fn new(config: DecisionConfig) -> Self {
        Self { config }
    }

    /// Build the candidate list from deterministic matches plus (optionally)
    /// the high-entropy backstop.
    ///
    /// `matches` is typically the output of `Detector::detect`, i.e. the
    /// regex+NER matches; the entropy backstop adds spans the deterministic
    /// layers did not identify (the blind spot).
    pub fn candidates(&self, text: &str, matches: &[PiiMatch]) -> Vec<Decision> {
        let mut out = Vec::new();

        // Deterministic (regex / NER) matches.
        for m in matches {
            out.push(Decision {
                start: Some(m.start),
                end: Some(m.end),
                text: m.matched_text.clone(),
                verdict: Verdict::Redact, // provisional; overwritten by model
                class: m.pattern_name.clone(),
                confidence: 1.0,
                reason: None,
                source: "detector".to_string(),
                pattern_name: Some(m.pattern_name.clone()),
                detector_confidence: Some(m.confidence),
            });
        }

        // High-entropy backstop for unlabeled secrets.
        if self.config.entropy_backstop {
            for c in entropy_candidates(text, matches) {
                out.push(Decision {
                    start: Some(c.start),
                    end: Some(c.end),
                    text: c.text.clone(),
                    verdict: Verdict::Flag,
                    class: "unlabeled".to_string(),
                    confidence: 0.0,
                    reason: None,
                    source: c.source,
                    pattern_name: c.pattern_name,
                    detector_confidence: c.detector_confidence,
                });
            }
        }

        if self.config.max_candidates > 0 && out.len() > self.config.max_candidates {
            out.truncate(self.config.max_candidates);
        }

        // De-duplicate overlapping spans (later model answers win for same
        // start/end), and sort by start.
        out.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| b.end.cmp(&a.end)));
        out.dedup_by(|a, b| a.start == b.start && a.end == b.end);
        out
    }

    /// Adjudicate a document. Returns a decision per candidate.
    ///
    /// Any candidate the model marks `keep` gets `verdict = Keep` (so a caller
    /// can veto a deterministic over-redaction); `redact` stays `Redact`; the
    /// model's `flag` / low-confidence answers become `Flag`.
    pub fn adjudicate(&self, text: &str, candidates: Vec<Decision>) -> Result<Vec<Decision>> {
        if !self.config.usable() {
            return Ok(candidates);
        }

        let mut out = Vec::new();
        for chunk in candidates.chunks(self.config.batch_size.max(1)) {
            if self.config.backend.eq_ignore_ascii_case("systemone") {
                // TypeSafe/Jev-compatible readout: send the whole chunk as
                // `state`, one Choice per candidate, over the /v1/systemone
                // endpoint. Returns true per-option probabilities and a
                // derived (calibrated) confidence.
                let answers = self.ask_systemone(text, chunk)?;
                for (i, cand) in chunk.iter().enumerate() {
                    let mut decision = cand.clone();
                    if let Some(ans) = answers.get(&i) {
                        apply_answer(&mut decision, ans, self.config.threshold);
                    } else {
                        decision.verdict = Verdict::Flag;
                        decision.reason = Some("no systemone answer".to_string());
                    }
                    out.push(decision);
                }
            } else {
                let prompt = self.build_prompt(text, chunk);
                let answers = self.ask(&prompt)?;
                for (i, cand) in chunk.iter().enumerate() {
                    let mut decision = cand.clone();
                    // Prefer the model's stated index when it supplies one;
                    // otherwise fall back to positional order. A candidate the
                    // model did not answer stays a `Flag` (unsure) rather than a
                    // hard redact/keep, so nothing is silently decided.
                    let ans = answers
                        .iter()
                        .find(|a| a.index == Some(i))
                        .or_else(|| answers.get(i));
                    if let Some(ans) = ans {
                        apply_answer(&mut decision, ans, self.config.threshold);
                    } else {
                        decision.verdict = Verdict::Flag;
                        decision.reason = Some("no model answer".to_string());
                    }
                    out.push(decision);
                }
            }
        }
        Ok(out)
    }

    /// Ask a TypeSafe/Jev-compatible `/v1/systemone` endpoint.
    ///
    /// Sends `state = <text>` with one `choice` question per candidate in the
    /// chunk (criteria: `redact` / `keep` / `flag`). The response's `answers`
    /// map is keyed by the question ids we sent, so we map back by index.
    /// This is the calibrated readout path: probabilities + derived confidence,
    /// unlike the chat backend's decoded-token confidence.
    fn ask_systemone(
        &self,
        text: &str,
        candidates: &[Decision],
    ) -> Result<std::collections::HashMap<usize, ModelAnswer>> {
        let mut questions = serde_json::Map::new();
        for (i, cand) in candidates.iter().enumerate() {
            let span = cand.text.clone();
            let ctx = context_around(text, cand, self.config.context_chars);
            let instructions = format!(
                "Adjudicate the span `{span}` (in the state) as a secret/credential/PII. \
                 Context: ...{ctx}..."
            );
            let mut criteria = serde_json::Map::new();
            criteria.insert(
                "redact".to_string(),
                serde_json::json!("Real secret/credential/PII; must be redacted"),
            );
            criteria.insert(
                "keep".to_string(),
                serde_json::json!("Benign code, path, uuid, checksum, or example; safe to keep"),
            );
            criteria.insert(
                "flag".to_string(),
                serde_json::json!("Uncertain; ambiguous identifiers, review manually"),
            );
            let q = serde_json::json!({
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            });
            questions.insert(format!("cand_{i}"), q);
        }
        let body = serde_json::json!({
            "state": text,
            "model": self.config.model,
            "questions": questions,
        });
        let body_str = serde_json::to_string(&body).context("serializing systemone request")?;
        let timeout = Duration::from_secs(self.config.timeout_secs.max(1));
        // The `endpoint` may be a base URL (e.g. http://host:8009) or already
        // include the path. Accept either; append /v1/systemone if needed.
        let base = self.config.endpoint.trim_end_matches('/');
        let url = if base.ends_with("/v1/systemone") || base.ends_with("/systemone") {
            base.to_string()
        } else {
            format!("{base}/v1/systemone")
        };
        let mut req = ureq::post(&url)
            .config()
            .timeout_global(Some(timeout))
            .build()
            .header("Content-Type", "application/json");
        if let Some(ref env_name) = self.config.api_key_env {
            let key = std::env::var(env_name)
                .map_err(|_| anyhow!("decision api_key_env '{env_name}' not set"))?;
            req = req.header("Authorization", &format!("Bearer {key}"));
        }
        let mut resp = req
            .send(body_str.as_str())
            .map_err(|e| anyhow!("systemone request failed: {e}"))?;
        let content = resp
            .body_mut()
            .read_to_string()
            .context("reading systemone response")?;
        if std::env::var("NYM_DEBUG_DECISION").is_ok() {
            eprintln!("[decision] systemone:");
            eprintln!("{}", serde_json::to_string_pretty(&body)?);
            eprintln!("[decision] systemone reply:\n{content}");
        }
        let parsed: SystemOneResponse =
            serde_json::from_str(&content).context("parsing systemone response")?;
        let mut out = std::collections::HashMap::new();
        for (i, _cand) in candidates.iter().enumerate() {
            let key = format!("cand_{i}");
            let ans = parsed.answers.get(&key);
            let ma = ans.and_then(SystemOneAnswer::as_decision);
            if let Some(ma) = ma {
                out.insert(i, ma);
            } else {
                out.insert(
                    i,
                    ModelAnswer {
                        index: Some(i),
                        verdict: "flag".to_string(),
                        class: "unlabeled".to_string(),
                        confidence: 0.0,
                        reason: Some("no answer".to_string()),
                        is_secret: None,
                    },
                );
            }
        }
        Ok(out)
    }

    fn build_prompt(&self, text: &str, candidates: &[Decision]) -> String {
        let mut lines = Vec::new();
        lines.push(
            "You adjudicate whether text spans are secrets/credentials/private data. \
             For each candidate, answer with a JSON object: {\"verdict\":\"redact|keep|flag\",\
             \"class\":\"<short class>\",\"confidence\":<0..1>}. \
             'redact'=real secret/credential/PII; 'keep'=benign (code sample, path, uuid, \
             checksum, example); 'flag'=unsure. Only output valid JSON, one object per line, \
             in the same order as the candidates."
                .to_string(),
        );
        lines.push("Candidates (index | span | context):".to_string());
        for (i, cand) in candidates.iter().enumerate() {
            let ctx = context_around(text, cand, self.config.context_chars);
            lines.push(format!(
                "[{i}] span='{}'\n   context=...{}...",
                cand.text, ctx
            ));
        }
        lines.join("\n")
    }

    /// Send one batched decision request and parse the per-candidate answers.
    fn ask(&self, prompt: &str) -> Result<Vec<ModelAnswer>> {
        // OpenAI chat-completions JSON mode is not guaranteed everywhere, so
        // we ask for JSON and parse the first JSON object per line defensively.
        let body = DecisionRequest {
            model: self.config.model.clone(),
            messages: vec![DecisionMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            temperature: 0.0,
            max_tokens: 1024,
            response_format: ResponseFormat {
                kind: "json_object".to_string(),
            },
        };

        let body_str = serde_json::to_string(&body).context("serializing decision request")?;
        let timeout = Duration::from_secs(self.config.timeout_secs.max(1));

        let mut req = ureq::post(&self.config.endpoint)
            .config()
            .timeout_global(Some(timeout))
            .build()
            .header("Content-Type", "application/json");
        if let Some(ref env_name) = self.config.api_key_env {
            let key = std::env::var(env_name)
                .map_err(|_| anyhow!("decision api_key_env '{env_name}' not set"))?;
            req = req.header("Authorization", &format!("Bearer {key}"));
        }

        let resp = req
            .send(body_str.as_str())
            .map_err(|e| anyhow!("decision request failed: {e}"))?;
        let mut resp = resp;
        let content = resp
            .body_mut()
            .read_to_string()
            .context("reading decision response")?;

        // The body is the chat-completions envelope; pull `.choices[0].content`
        // and parse THAT as the candidate answers.
        let inner = serde_json::from_str::<ChatResponse>(&content)
            .ok()
            .and_then(|c| c.choices.into_iter().next())
            .map_or_else(|| content.clone(), |c| c.message.content);
        if std::env::var("NYM_DEBUG_DECISION").is_ok() {
            eprintln!("[decision] inner:\n{inner}");
        }

        Ok(parse_model_answer_lines(&inner))
    }
}

/// Extract named-model punctures: apply a raw model answer onto a decision.
fn apply_answer(decision: &mut Decision, ans: &ModelAnswer, threshold: f32) {
    decision.class.clone_from(&ans.class);
    decision.confidence = ans.confidence;
    decision.reason.clone_from(&ans.reason);

    let v = ans.verdict.to_lowercase();
    // Trust the model's explicit verdict if it is unambiguous.
    if v == "keep" || v == "benign" {
        decision.verdict = Verdict::Keep;
    } else if v == "flag" || v == "unknown" || v == "unsure" {
        decision.verdict = Verdict::Flag;
    } else if v == "redact" || v == "secret" || v == "private" {
        decision.verdict = Verdict::Redact;
    } else {
        // Fall back to the confidence cut-off: a high-confidence "secret"
        // confidence means redact, low means keep.
        let c = ans.confidence.clamp(0.0, 1.0);
        decision.verdict = if c >= threshold {
            Verdict::Redact
        } else {
            Verdict::Keep
        };
    }
    // A low-confidence model answer should be a flag, not a hard verdict.
    if decision.verdict != Verdict::Keep && ans.confidence < threshold * 0.6 {
        decision.verdict = Verdict::Flag;
    }
}

/// Parse the model's output into one answer per candidate.
///
/// Handles three shapes a decision model might return:
/// - one JSON object per line (the requested form);
/// - a single JSON object (the model answered once, e.g. for one candidate);
/// - a JSON array of objects.
///
/// Any prose wrapping is tolerated by scanning for `{...}` blocks.
fn parse_model_answer_lines(content: &str) -> Vec<ModelAnswer> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    // Try a JSON array first.
    if let Ok(v) = serde_json::from_str::<Vec<ModelAnswer>>(trimmed) {
        return v;
    }

    // Otherwise scan for brace-delimited objects in order (covers one-per-line,
    // single object, and objects glued together).
    let mut out = Vec::new();
    let bytes = trimmed.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Find the next '{'.
        let open = match bytes[i..].iter().position(|&b| b == b'{') {
            Some(p) => i + p,
            None => break,
        };
        // Match the closing brace (naive nesting depth).
        let mut depth = 0usize;
        let mut close = open;
        for (offset, &b) in bytes[open..].iter().enumerate() {
            match b {
                b'{' => depth += 1,
                b'}' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        close = open + offset;
                        break;
                    }
                }
                _ => {}
            }
        }
        let candidate = &trimmed[open..=close];
        if let Ok(ans) = serde_json::from_str::<ModelAnswer>(candidate) {
            out.push(ans);
        }
        i = close + 1;
    }
    out
}

/// Return a bounded window of surrounding context for a candidate.
fn context_around(text: &str, cand: &Decision, context_chars: usize) -> String {
    let start = cand.start.unwrap_or(0).min(text.len());
    let end = cand.end.unwrap_or(start).min(text.len());
    let ctx_start = start.saturating_sub(context_chars);
    let ctx_end = (end + context_chars).min(text.len());
    let s = text.get(ctx_start..ctx_end).unwrap_or_default();
    s.replace('\n', " ")
}

/// High-entropy (base64/hex-like) candidate spans that the deterministic
/// layers did not already cover -- the unlabeled-secret backstop.
fn entropy_candidates(text: &str, matches: &[PiiMatch]) -> Vec<Candidate> {
    use std::collections::HashSet;

    // Build the set of spans already claimed by regex/NER so we don't re-flag.
    let claimed: HashSet<(usize, usize)> = matches.iter().map(|m| (m.start, m.end)).collect();

    let mut out = Vec::new();
    // Reuse nym's token scan: a run of base64/hex alphabet chars, length >= 32.
    for m in TOKEN_ISH.find_iter(text) {
        let s = m.as_str();
        if s.len() < 32 {
            continue;
        }
        let (start, end) = (m.start(), m.end());
        if claimed.iter().any(|(cs, ce)| start < *ce && end > *cs) {
            continue;
        }
        // Only flag if entropy is high (not a plain path/uuid/checksum).
        if !high_entropy(s) {
            continue;
        }
        out.push(Candidate {
            start,
            end,
            text: s.to_string(),
            source: "entropy".to_string(),
            pattern_name: None,
            detector_confidence: None,
        });
    }
    out
}

#[expect(
    clippy::cast_precision_loss,
    reason = "character-count entropy of a token; usize->f64 for log2 division cannot lose meaningful precision at these magnitudes"
)]
fn high_entropy(s: &str) -> bool {
    use std::collections::HashMap;
    if s.len() < 32 {
        return false;
    }
    let mut counts: HashMap<char, usize> = HashMap::new();
    for c in s.chars() {
        *counts.entry(c).or_insert(0) += 1;
    }
    let n = s.len() as f64;
    let h: f64 = counts
        .values()
        .map(|&v| {
            let p = v as f64 / n;
            -p * p.log2()
        })
        .sum();
    h > 4.0
}

static TOKEN_ISH: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    #[expect(clippy::expect_used, reason = "static regex literal is infallible")]
    regex::Regex::new(r"\b[A-Za-z0-9+/=_\-]{32,}\b").expect("static regex is valid")
});
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn ans(verdict: &str, class: &str, conf: f32) -> ModelAnswer {
        ModelAnswer {
            index: None,
            verdict: verdict.into(),
            class: class.into(),
            confidence: conf,
            reason: None,
            is_secret: None,
        }
    }

    #[test]
    fn parses_single_object() {
        let a = parse_model_answer_lines(
            "{\"verdict\":\"redact\",\"class\":\"api_key\",\"confidence\":0.9}",
        );
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].verdict, "redact");
        assert_eq!(a[0].confidence, 0.9);
    }

    #[test]
    fn parses_one_per_line() {
        let c = "{\"verdict\":\"redact\",\"class\":\"a\",\"confidence\":0.9}\n{\"verdict\":\"keep\",\"class\":\"b\",\"confidence\":0.8}";
        let a = parse_model_answer_lines(c);
        assert_eq!(a.len(), 2);
        assert_eq!(a[1].verdict, "keep");
    }

    #[test]
    fn parses_array() {
        let c = "[{\"verdict\":\"redact\",\"class\":\"a\",\"confidence\":1.0},{\"verdict\":\"keep\",\"class\":\"b\",\"confidence\":0.9}]";
        let a = parse_model_answer_lines(c);
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn parses_prose_wrapped_object() {
        let c = "Sure! Here is my judgment: {\"verdict\":\"flag\",\"class\":\"unknown\",\"confidence\":0.4}. Hope this helps.";
        let a = parse_model_answer_lines(c);
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].verdict, "flag");
    }

    #[test]
    fn empty_content_yields_no_answers() {
        assert!(parse_model_answer_lines("").is_empty());
        assert!(parse_model_answer_lines("no JSON here").is_empty());
    }

    #[test]
    fn apply_answer_maps_verdicts() {
        let mut d = Decision {
            start: Some(0),
            end: Some(4),
            text: "test".into(),
            verdict: Verdict::Redact,
            class: "x".into(),
            confidence: 1.0,
            reason: None,
            source: "detector".into(),
            pattern_name: None,
            detector_confidence: None,
        };
        apply_answer(&mut d, &ans("keep", "example", 0.9), 0.5);
        assert_eq!(d.verdict, Verdict::Keep);
        assert_eq!(d.class, "example");
    }

    #[test]
    fn systemone_choice_maps_verdict_and_confidence() {
        // A real Jev/Kev `/v1/systemone` choice answer.
        let a = SystemOneAnswer {
            qtype: "choice".into(),
            choice: Some("redact".into()),
            noul: None,
            score: None,
            confidence: Some(0.76),
            probabilities: vec![
                ("redact".to_string(), 0.84),
                ("keep".to_string(), 0.03),
                ("flag".to_string(), 0.12),
            ]
            .into_iter()
            .collect(),
        };
        let ma = a.as_decision().expect("choice answer should map");
        assert_eq!(ma.verdict, "redact");
        assert_eq!(ma.confidence, 0.76);
        // Class becomes the most-probable option (the chosen one here).
        assert_eq!(ma.class, "redact");
    }

    #[test]
    fn systemone_noul_thresholds_to_verdict() {
        let yes = SystemOneAnswer {
            qtype: "noul".into(),
            choice: None,
            noul: Some(0.9),
            score: None,
            confidence: None,
            probabilities: HashMap::new(),
        };
        let ma = yes.as_decision().expect("noul answer should map");
        assert_eq!(ma.verdict, "redact");
        assert_eq!(ma.confidence, 0.9);

        let no = SystemOneAnswer {
            qtype: "noul".into(),
            choice: None,
            noul: Some(0.2),
            score: None,
            confidence: None,
            probabilities: HashMap::new(),
        };
        let ma2 = no.as_decision().unwrap();
        assert_eq!(ma2.verdict, "keep");
    }

    #[test]
    fn systemone_unknown_type_yields_none() {
        let a = SystemOneAnswer {
            qtype: "score".into(),
            choice: None,
            noul: None,
            score: Some(1.37),
            confidence: Some(0.34),
            probabilities: HashMap::new(),
        };
        assert!(a.as_decision().is_none());
    }

    #[test]
    fn systemone_response_parses_answers_by_question_id() {
        let raw = r#"{"model":"kev-latest","answers":{
            "cand_0":{"type":"choice","choice":"redact","confidence":0.76,
                       "probabilities":{"redact":0.84,"keep":0.03,"flag":0.12}},
            "cand_1":{"type":"choice","choice":"keep","confidence":0.9,
                       "probabilities":{"redact":0.02,"keep":0.93,"flag":0.05}}
        },"usage":{"input_tokens":227,"output_tokens":179}}"#;
        let resp: SystemOneResponse = serde_json::from_str(raw).expect("parse");
        assert!(resp.answers.contains_key("cand_0"));
        let a0 = resp.answers["cand_0"].as_decision().unwrap();
        assert_eq!(a0.verdict, "redact");
        let a1 = resp.answers["cand_1"].as_decision().unwrap();
        assert_eq!(a1.verdict, "keep");
    }

    #[test]
    fn apply_answer_low_confidence_becomes_flag() {
        let mut d = Decision {
            start: Some(0),
            end: Some(4),
            text: "test".into(),
            verdict: Verdict::Redact,
            class: "x".into(),
            confidence: 1.0,
            reason: None,
            source: "detector".into(),
            pattern_name: None,
            detector_confidence: None,
        };
        // Model says redact but at 0.2 confidence (< 0.5*0.6) -> flag.
        apply_answer(&mut d, &ans("redact", "secret", 0.2), 0.5);
        assert_eq!(d.verdict, Verdict::Flag);
    }

    #[test]
    fn entropy_backstop_flags_unlabeled() {
        // A high-entropy 32+ char base64 blob with no name and no known prefix.
        let text = "token abcDEF0123456789abcdefghijklmnopqrstuvwxyzXYZ xyz";
        let m: Vec<PiiMatch> = Vec::new();
        let cands = entropy_candidates(text, &m);
        assert!(!cands.is_empty());
        assert_eq!(cands[0].source, "entropy");
    }

    #[test]
    fn entropy_backstop_skips_claimed_span() {
        let text = "abcDEF0123456789abcdefghijklmnopqrstuvwxyzXYZ";
        let m = vec![PiiMatch {
            pattern_name: "api_key".into(),
            matched_text: text.into(),
            start: 0,
            end: text.len(),
            confidence: Confidence::High,
            category: super::super::patterns::PiiCategory::Authentication,
        }];
        let cands = entropy_candidates(text, &m);
        assert!(cands.is_empty());
    }
}
