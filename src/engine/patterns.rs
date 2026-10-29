//! PII detection patterns.
//!
//! This module defines all built-in regex patterns for detecting
//! personally identifiable information (PII).

use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

/// Compile a static regex from a string literal.
///
/// This wraps `Regex::new(...).unwrap()` for use in `LazyLock` static initializers.
/// The unwrap is safe because all patterns are compile-time string literals
/// that have been validated by tests. Any failure here would be a programmer
/// error caught at first use.
#[expect(clippy::unwrap_used, reason = "Static regex literals are infallible")]
fn regex(pattern: &str) -> Regex {
    Regex::new(pattern).unwrap()
}

/// Confidence level for a PII pattern match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Confidence {
    /// Low confidence - loose patterns, context-dependent
    Low,
    /// Medium confidence - common patterns that may have false positives
    Medium,
    /// High confidence - structured patterns with checksums or strict formats
    High,
}

/// Category of PII type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PiiCategory {
    /// Contact information (email, phone)
    Contact,
    /// Identity documents (SSN, passport)
    Identity,
    /// Financial information (credit card, IBAN)
    Financial,
    /// Network identifiers (IP, MAC)
    Network,
    /// Authentication tokens (API keys, JWT)
    Authentication,
    /// Social media handles and usernames
    Social,
    /// Other identifiers (UUID)
    Other,
}

/// A PII pattern definition.
#[derive(Debug, Clone)]
pub struct PiiPattern {
    /// Unique identifier for this pattern
    pub name: &'static str,
    /// Human-readable description
    pub description: &'static str,
    /// The compiled regex pattern
    pub regex: &'static LazyLock<Regex>,
    /// Confidence level
    pub confidence: Confidence,
    /// Category of PII
    pub category: PiiCategory,
    /// Example of what this pattern matches
    pub example: &'static str,
    /// Default replacement placeholder for this pattern type.
    /// Used by the placeholder replacement strategy.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Public API - used by consumers")
    )]
    pub replacement_template: &'static str,
}

// =============================================================================
// Pattern Definitions
// =============================================================================

// Email pattern
static EMAIL_REGEX: LazyLock<Regex> =
    LazyLock::new(|| regex(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b"));

// US Phone number patterns
static PHONE_US_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    // Matches: (555) 123-4567, 555-123-4567, 555.123.4567, 5551234567, +1 555 123 4567
    regex(r"(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?[2-9]\d{2}[-.\s]?\d{4}")
});

// International phone (E.164 format, with optional spaces/separators)
static PHONE_INTL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    // Matches: +49192836418, +49 192836418, +49 192 836 418, +49-192-836-418
    regex(r"\+[1-9][\d\s\-]{6,17}\d\b")
});

// US Social Security Number (with dashes, dots, or spaces)
static SSN_REGEX: LazyLock<Regex> = LazyLock::new(|| regex(r"\b\d{3}[-.\s]\d{2,3}[-.\s]\d{4}\b"));

// SSN without dashes (9 consecutive digits with word boundary)
static SSN_NODASH_REGEX: LazyLock<Regex> = LazyLock::new(|| regex(r"\b\d{9}\b"));

// Credit card numbers (major brands)
static CREDIT_CARD_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    regex(
        r"(?x)
        \b
        (?:
            4\d{3}|                    # Visa
            5[1-5]\d{2}|               # Mastercard
            3[47]\d{2}|                # Amex
            6(?:011|5\d{2})            # Discover
        )
        [-\s]?
        \d{4}[-\s]?\d{4}[-\s]?\d{4}
        \b
        ",
    )
});

// Credit card without separators
static CREDIT_CARD_NODASH_REGEX: LazyLock<Regex> =
    LazyLock::new(|| regex(r"\b(?:4\d{15}|5[1-5]\d{14}|3[47]\d{13}|6(?:011|5\d{2})\d{12})\b"));

// IPv4 address
/// True when an IPv4 address carries no personal information: loopback, RFC1918
/// private space, link-local, CGNAT (100.64.0.0/10, which includes the tailnet),
/// multicast/reserved blocks and the documentation/benchmark ranges. These appear
/// constantly in agent logs as infrastructure and were the largest measured
/// false-positive family after URLs (70% of reviewed ipv4 findings were harmless).
pub(crate) fn is_local_ipv4(value: &str) -> bool {
    let octets: Option<Vec<u8>> = value
        .split('.')
        .map(|part| part.parse::<u8>().ok())
        .collect();
    let Some(octets) = octets.filter(|octets| octets.len() == 4) else {
        return false;
    };
    let [a, b, c, _] = [octets[0], octets[1], octets[2], octets[3]];
    matches!(a, 0 | 10 | 127)                 // this-network, RFC1918, loopback
        || (a == 172 && (16..=31).contains(&b))   // RFC1918
        || (a == 192 && b == 168)                 // RFC1918
        || (a == 169 && b == 254)                 // link-local
        || (a == 100 && (64..=127).contains(&b))  // CGNAT / tailnet
        || a >= 224                               // multicast, reserved, broadcast
        || (a == 192 && b == 0 && c == 2)         // TEST-NET-1
        || (a == 198 && b == 51 && c == 100)      // TEST-NET-2
        || (a == 203 && b == 0 && c == 113)       // TEST-NET-3
        || (a == 198 && (b == 18 || b == 19)) // benchmarking
}

static IPV4_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    regex(
        r"(?x)
        \b
        (?:
            (?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.
        ){3}
        (?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)
        \b
        ",
    )
});

// IPv6 address (simplified - matches common formats)
static IPV6_REGEX: LazyLock<Regex> =
    LazyLock::new(|| regex(r"(?i)\b(?:[a-f0-9]{1,4}:){7}[a-f0-9]{1,4}\b"));

// MAC address
static MAC_REGEX: LazyLock<Regex> =
    LazyLock::new(|| regex(r"(?i)\b(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}\b"));

// UUID
static UUID_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    regex(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
});

// JWT token
static JWT_REGEX: LazyLock<Regex> =
    LazyLock::new(|| regex(r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*"));

// AWS Access Key ID
static AWS_KEY_REGEX: LazyLock<Regex> =
    LazyLock::new(|| regex(r"(?:A3T[A-Z0-9]|AKIA|AGPA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}"));

// Generic API key pattern (40+ char alphanumeric)
static API_KEY_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    regex(r"\b(?:sk|pk|api|key|token)[-_]?(?:live|test|prod)?[-_]?[a-zA-Z0-9]{32,}\b")
});

// Vendor-prefixed credentials. The prefixes are distinctive, so these are
// high confidence, unlike the generic `api_key` shape. Internal hyphens and
// underscores (sk-ant-api03-..., sk-proj-..., github_pat_...) are part of the
// token and must be redacted with it.
static VENDOR_SECRET_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    regex(
        r"(?x)
        \b(?:
          sk-ant-[a-z]+\d{2}-[A-Za-z0-9_-]{20,}
        | sk-(?:proj|svcacct|admin|or-v1)-[A-Za-z0-9_-]{20,}
        | sk-[A-Za-z0-9]{32,}
        | gh[pousr]_[A-Za-z0-9]{36,}
        | github_pat_[A-Za-z0-9_]{40,}
        | glpat-[A-Za-z0-9_-]{20,}
        | xox[abposr]-[A-Za-z0-9-]{10,}
        | AIza[0-9A-Za-z_-]{35}
        | hf_[A-Za-z0-9]{30,}
        | npm_[A-Za-z0-9]{36}
        | pypi-[A-Za-z0-9_-]{50,}
        | (?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{20,}
        | whsec_[A-Za-z0-9+/=]{20,}
        | SG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}
        | dop_v1_[a-f0-9]{64}
        | shp(?:at|ss|ca)_[a-fA-F0-9]{32}
        | tskey-[A-Za-z0-9-]{20,}
        | lin_api_[A-Za-z0-9]{30,}
        | gsk_[A-Za-z0-9]{40,}
        )",
    )
});

// Environment-style secret assignments (.env files, shell exports). Upper-case
// Environment/config-style secret assignments. Two shapes:
// - SCREAMING_CASE keys accept bare or quoted values with `=` or `:`
//   (configuration identifiers; variable names keep this to configuration,
//   not ordinary code identifiers; shell expansions (`$VAR`) are references,
//   not values).
// - Any-case keys (optionally JSON/YAML-quoted) accept only quoted literal
//   values: `password="..."`, `"password": "..."`, `token: '...'`. Quoting
//   marks a literal, so ordinary code identifiers (`password = formData.value`)
//   and bare lowercase assignments stay unmatched.
static SECRET_ASSIGNMENT_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    regex(
        r#"\b[A-Z][A-Z0-9_]*(?:PASSWORD|PASSWD|SECRET|TOKEN|API_KEY|APIKEY|ACCESS_KEY|PRIVATE_KEY)[A-Z0-9_]*\s*[:=]\s*["']?(?P<pii>[^\s"'`$][^\s"'`]{7,})|(?i:["']?(?:password|passwd|secret|token|api_?key|access_?key|private_?key)["']?[ \t]*[:=][ \t]*["'](?P<piz>[^\s"'`$][^\s"'`]{7,})["'])"#,
    )
});

// Credentials embedded in a URL: only the password of `scheme://user:password@`
// is the finding; scheme, user and host stay readable.
static URL_CREDENTIALS_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    regex(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://(?P<user>[^\s/:@]+):(?P<pii>[^\s/@]{3,})@")
});

/// Schemes of references to a secret held elsewhere (config, keychain, vault).
const SECRET_REFERENCE_SCHEMES: &[&str] = &[
    "env:",
    "keychain:",
    "kyz:",
    "vault:",
    "op://",
    "file:",
    "secret:",
    "secrets:",
    "ssm:",
    "sops:",
    "pass:",
];

/// Marker words that make an assigned value an obvious fixture or placeholder.
const PLACEHOLDER_WORDS: &[&str] = &[
    "synthetic",
    "fixture",
    "test",
    "example",
    "dummy",
    "fake",
    "sample",
    "placeholder",
    "sentinel",
    "changeme",
    "redacted",
    "your",
    "mock",
];

/// Well-known default or demo passwords in connection URLs.
const DEFAULT_URL_PASSWORDS: &[&str] = &[
    "password",
    "pass",
    "passwd",
    "secret",
    "postgres",
    "root",
    "admin",
    "guest",
    "changeme",
    "example",
    "test",
    "user",
    "mysql",
    "redis",
    "minio",
    "minioadmin",
];

/// A segment that could be key material: 8+ characters mixing letters and digits.
fn random_looking(segment: &str) -> bool {
    segment.len() >= 8
        && segment.bytes().any(|b| b.is_ascii_digit())
        && segment.bytes().any(|b| b.is_ascii_alphabetic())
}

/// A value assigned to a secret-named key that is not itself a secret: a
/// reference to one (`env:OPENAI_API_KEY`, `keychain:name`, `${DB_PASS}`,
/// `op://vault/item`), an environment lookup, a mask (`xxxx`, `****`), or a
/// value that names itself a fixture (`synthetic-access`, `your-api-key`)
/// without any segment that could be real key material. Plain words stay
/// findings: `correct-horse-battery` is a valid passphrase.
pub(crate) fn non_secret_value(value: &str) -> bool {
    let v = value.trim_matches(|c: char| c == '"' || c == '\'' || c == '`');
    let lower = v.to_ascii_lowercase();
    if SECRET_REFERENCE_SCHEMES
        .iter()
        .any(|s| lower.starts_with(s))
        || v.starts_with("${")
        || v.starts_with("{{")
        || (v.starts_with('<') && v.ends_with('>'))
        || (v.len() > 2 && v.starts_with('%') && v.ends_with('%'))
        || lower.starts_with("process.env.")
        || lower.starts_with("os.environ")
        || lower.contains("getenv(")
        || lower.starts_with("env[")
    {
        return true;
    }
    let masked = v
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>();
    if !v.is_empty()
        && (v.chars().all(|c| matches!(c, '*' | '.' | '-' | '_' | '#'))
            || (masked.len() >= 3 && masked.chars().all(|c| c == 'x' || c == 'X')))
    {
        return true;
    }
    let segments: Vec<&str> = v
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|s| !s.is_empty())
        .collect();
    let marked = segments.iter().any(|s| {
        let s = s.to_ascii_lowercase();
        PLACEHOLDER_WORDS.contains(&s.as_str()) || (s.len() >= 3 && s.chars().all(|c| c == 'x'))
    });
    marked && !segments.iter().any(|s| random_looking(s))
}

/// A URL password that is a well-known default, equals its user, or is a
/// reference or placeholder: not a credential worth reporting.
pub(crate) fn non_secret_url_password(user: &str, password: &str) -> bool {
    password.eq_ignore_ascii_case(user)
        || DEFAULT_URL_PASSWORDS.contains(&password.to_ascii_lowercase().as_str())
        || non_secret_value(password)
}

/// File extensions that end asset names written like addresses (`logo@3x.png`).
const FILE_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "svg", "webp", "ico", "bmp", "avif", "mp4", "mov", "webm", "mp3",
    "wav", "pdf", "md", "txt", "json", "js", "ts", "tsx", "css", "html", "lock",
];

/// Address-shaped text that is no address: an SSH remote (`git@github.com`)
/// or an asset file name (`logo@3x.png`, `intro@2x.mp4`).
pub(crate) fn non_address_email(value: &str) -> bool {
    let Some((local, domain)) = value.rsplit_once('@') else {
        return false;
    };
    let tld = domain
        .rsplit('.')
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    local == "git" || FILE_EXTENSIONS.contains(&tld.as_str())
}

/// Character set of opaque identifiers: ids, hashes, base64 blobs and the
/// separators used inside them. No '/' or '_': a path or a snake_case name is
/// not an opaque id, and that is where a person's name or account sits.
fn opaque_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '+' | '=')
}

/// The run of identifier characters covering the span, when it is long and not
/// purely digits: a UUID, a hex id or digest, a base64 payload, a lockfile hash.
fn opaque_run(text: &str, start: usize, end: usize) -> Option<(usize, usize)> {
    let run_start = text[..start]
        .char_indices()
        .rev()
        .take_while(|(_, c)| opaque_char(*c))
        .last()
        .map_or(start, |(i, _)| i);
    let run_end = text[end..]
        .char_indices()
        .take_while(|(_, c)| opaque_char(*c))
        .last()
        .map_or(end, |(i, c)| end + i + c.len_utf8());
    let run = &text[run_start..run_end];
    (run.len() >= 20 && run.chars().any(|c| c.is_ascii_alphabetic()))
        .then_some((run_start, run_end))
}

/// The span is a fragment of an opaque run, not a token of its own.
pub(crate) fn strictly_inside_opaque(text: &str, start: usize, end: usize) -> bool {
    opaque_run(text, start, end).is_some_and(|(a, b)| a < start || b > end)
}

/// The match sits in a value assigned to a credential-named key, where an id or
/// a hash really is the secret (`api_key: <hex>`, `refresh_token=<hex>`).
pub(crate) fn in_credential_field(text: &str, start: usize) -> bool {
    // The window must start on a character boundary: text is UTF-8.
    let mut window_start = start.saturating_sub(48);
    while !text.is_char_boundary(window_start) {
        window_start += 1;
    }
    let window = &text[window_start..start];
    let Some(cut) = window.rfind(['=', ':', '"', '\'', ' ', ',']) else {
        return false;
    };
    let key =
        window[..cut].trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-');
    let key = key.to_ascii_lowercase();
    [
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "access_key",
        "private_key",
        "credential",
    ]
    .iter()
    .any(|name| key.ends_with(name))
}

/// Text that is unambiguously code syntax rather than a name: dunder and MCP
/// names (`mcp__linear__save_issue`), CLI flags, `::` paths, file paths, dotted
/// module paths, SCREAMING_CASE constants. Deliberately excludes plain snake_case:
/// project codenames and usernames look exactly like `max_retries`, and the owner's
/// labels show suppressing those costs real names for a handful of false
/// positives (identifier rule measured on the fresh round: -7 harmless, -2 real).
pub(crate) fn identifier_like(value: &str) -> bool {
    let v = value.trim();
    if v.is_empty() || v.contains(char::is_whitespace) {
        return false;
    }
    if v.starts_with("--") || v.contains("::") || v.contains("__") {
        return true;
    }
    if v.contains('/') || v.contains('\\') {
        return true;
    }
    let underscore = v.contains('_');
    let screaming = underscore && v.chars().all(|c| !c.is_lowercase());
    if screaming {
        return true;
    }
    // A dotted name with at least two dots and no spaces is a module path.
    let dots = v.chars().filter(|c| *c == '.').count();
    dots >= 2 && v.split('.').all(|part| !part.is_empty())
}

// PEM private-key blocks, including the armor lines.
static PRIVATE_KEY_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    regex(r"(?s)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----")
});

// IBAN candidates, compact or in space-separated groups of four. Only spans that
// pass `iban_span` (country length and ISO 13616 mod-97) become findings.
static IBAN_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    regex(r"\b[A-Z]{2}\d{2}(?:[A-Z0-9]{11,30}|(?: [A-Z0-9]{4}){2,7}(?: [A-Z0-9]{1,4})?)\b")
});

const IBAN_LENGTHS: &[(&str, usize)] = &[
    ("AD", 24),
    ("AE", 23),
    ("AT", 20),
    ("BE", 16),
    ("BG", 22),
    ("CH", 21),
    ("CY", 28),
    ("CZ", 24),
    ("DE", 22),
    ("DK", 18),
    ("EE", 20),
    ("ES", 24),
    ("FI", 18),
    ("FR", 27),
    ("GB", 22),
    ("GI", 23),
    ("GR", 27),
    ("HR", 21),
    ("HU", 28),
    ("IE", 22),
    ("IS", 26),
    ("IT", 27),
    ("LI", 21),
    ("LT", 20),
    ("LU", 20),
    ("LV", 21),
    ("MC", 27),
    ("MT", 31),
    ("NL", 18),
    ("NO", 15),
    ("PL", 28),
    ("PT", 25),
    ("RO", 24),
    ("SE", 24),
    ("SI", 19),
    ("SK", 24),
    ("SM", 27),
    ("TR", 26),
    ("VA", 22),
];

/// Validate an IBAN candidate match and return the exact IBAN byte range.
///
/// Known countries must have their registered length; a spaced candidate that
/// swallowed a following word is trimmed to that length at a token boundary.
/// Unknown countries are accepted only when the whole candidate validates.
pub(crate) fn iban_span(text: &str, start: usize, end: usize) -> Option<std::ops::Range<usize>> {
    let candidate = &text[start..end];
    let expected = IBAN_LENGTHS
        .iter()
        .find(|(country, _)| candidate.starts_with(country))
        .map(|&(_, length)| length);
    let mut compact = String::new();
    let mut span_end = start;
    for (offset, c) in candidate.char_indices() {
        if c == ' ' {
            continue;
        }
        compact.push(c);
        span_end = start + offset + c.len_utf8();
        if Some(compact.len()) == expected {
            break;
        }
    }
    let length_ok = expected.map_or((15..=34).contains(&compact.len()), |n| compact.len() == n);
    let boundary_ok = !text[span_end..]
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphanumeric());
    (length_ok && boundary_ok && iban_checksum_valid(&compact)).then_some(start..span_end)
}

fn iban_checksum_valid(iban: &str) -> bool {
    let mut remainder = 0u32;
    for c in iban[4..].chars().chain(iban[..4].chars()) {
        let Some(value) = c.to_digit(36) else {
            return false;
        };
        remainder = (remainder * if value < 10 { 10 } else { 100 } + value) % 97;
    }
    remainder == 1
}

// US Passport
static PASSPORT_US_REGEX: LazyLock<Regex> = LazyLock::new(|| regex(r"\b[A-Z]\d{8}\b"));

// Date patterns (various formats)
// ISO: 2024-12-25, US: 12/25/2024, EU: 25.12.2024, etc.
static DATE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    regex(
        r"(?x)
        \b
        (?:
            # ISO format: 2024-12-25
            \d{4}[-/]\d{1,2}[-/]\d{1,2} |
            # US/EU format: 12/25/2024 or 25.12.2024
            \d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4} |
            # Written: December 25, 2024 or 25 December 2024
            (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4} |
            \d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{2,4}
        )
        \b
        ",
    )
});

// Time patterns: 14:30, 2:30 PM, 10:20am, 7 o'clock, quarter past 13, etc.
static TIME_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    regex(
        r"(?ix)
        \b
        (?:
            # Standard time with AM/PM (no space): 10:20am, 2:30PM
            (?:0?[1-9]|1[0-2]):[0-5]\d\s?(?:AM|PM) |
            # 24-hour time: 14:30, 21:45
            (?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)? |
            # Written time: 7 o'clock, quarter past 13, half past 16
            (?:quarter|half)\s+(?:past|to)\s+\d{1,2} |
            \d{1,2}\s+o'clock |
            # Hour with AM/PM: 3 AM, 7pm
            \d{1,2}\s?(?:AM|PM)
        )
        \b
        ",
    )
});

// Bare numeric/underscore tokens are usually code, not evidence of an account.
// Keep conventional userNNN identifiers; require an explicit account label for
// other usernames. The named `pii` capture narrows contextual matches to the
// account value, preserving field labels, separators, and surrounding quotes.
static USERNAME_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    regex(
        r#"(?x)
        \buser\d{1,20}\b |
        ["']?\b(?i:username|user_name|login|handle)["']?\s*[:=]\s*["']?
        (?P<pii>[a-zA-Z][a-zA-Z0-9_.-]{2,31}\b)["']?
        "#,
    )
});

// Home-directory paths: /Users/<user>/..., /home/<user>/..., /Volumes/<name>/...,
// and Windows C:\\Users\\<user>\\... (or forward slashes C:/Users/...).
// These leak a real local username (a person's account) plus machine layout, and
// are a top sensitive surface in agent transcripts. High confidence: a leading
// /Users/ or /home/ is unambiguous. The username may contain non-ASCII letters
// /marks (müller, 東京) and Windows usernames; \p{L}\p{N}\p{M} covers letters,
// numbers and combining diacritics, so accented names redact in full.
static HOME_DIR_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    regex(r"(?i)(?:(?:/|\\)(?:users|home|volumes)|[a-z]:[/\\]users)[/\\][\p{L}\p{N}\p{M}._-]+")
});

// Unix system paths: /etc/..., /var/..., /opt/..., /usr/..., /srv/..., /root/...
// Reveal internal infrastructure layout (host config, launch agents, deployment
// paths). High confidence; the decision gate adjudicates redact/keep. Segment
// chars accept Unicode letters/marks so hostnames and config names in any
// script are captured whole.
static UNIX_PATH_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    regex(
        r"(?i)/(?:etc|var|opt|usr|srv|root|dev|proc|sys|lib|bin|sbin|tmp|mnt|media)/[\p{L}\p{N}\p{M}._-]+",
    )
});

// Social media handles
// Twitter/X handle: @username (1-15 chars, alphanumeric + underscore)
// Note: We exclude emails by requiring @ to NOT be preceded by alphanumeric
// This is handled in post-processing since regex crate doesn't support look-behind
static TWITTER_HANDLE_REGEX: LazyLock<Regex> =
    LazyLock::new(|| regex(r"@[a-zA-Z_][a-zA-Z0-9_]{0,14}\b"));

// Generic social handle (covers most platforms): @username
// More permissive than platform-specific patterns
static SOCIAL_HANDLE_REGEX: LazyLock<Regex> =
    LazyLock::new(|| regex(r"@[a-zA-Z][a-zA-Z0-9_.]{1,30}\b"));

// URL with username path (e.g., twitter.com/username, github.com/username)
static SOCIAL_URL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    regex(
        r"(?i)https?://(?:www\.)?(?:twitter|x|instagram|facebook|fb|linkedin|github|tiktok|youtube|reddit)\.com/@?[a-zA-Z0-9_.-]{1,39}(?:\?|/|$)",
    )
});

// GPS coordinates (latitude, longitude)
// Must have decimal points to differentiate from other number pairs
// Matches: 40.7128,-74.0060, 40.7128, -74.0060, (40.7128, -74.0060)
static GEOCOORD_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    regex(
        r"(?x)
        \(?
        [-+]?(?:[1-8]?\d\.\d{2,}|90\.0+)  # Latitude: -90 to 90, MUST have decimal with 2+ digits
        \s*[,]\s*                          # Comma separator (required for specificity)
        [-+]?(?:1[0-7]\d\.\d{2,}|180\.0+|\d{1,2}\.\d{2,})  # Longitude: -180 to 180
        \)?
        ",
    )
});

// European Social/Tax ID patterns
// German Tax ID (Steueridentifikationsnummer): 11 digits
// Note: Not added to BUILTIN_PATTERNS due to high false positive rate with generic 11-digit numbers
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "Public API - used by consumers")
)]
static DE_TAX_ID_REGEX: LazyLock<Regex> = LazyLock::new(|| regex(r"\b\d{11}\b"));

// UK National Insurance Number: 2 letters, 6 digits, 1 letter
static UK_NINO_REGEX: LazyLock<Regex> =
    LazyLock::new(|| regex(r"(?i)\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b"));

// French Social Security Number (NIR): 13 digits + 2 digit key
static FR_NIR_REGEX: LazyLock<Regex> =
    LazyLock::new(|| regex(r"\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b"));

// Italian Fiscal Code (Codice Fiscale): 16 alphanumeric chars
static IT_CF_REGEX: LazyLock<Regex> =
    LazyLock::new(|| regex(r"(?i)\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b"));

// Spanish DNI/NIE: 8 digits + letter or X/Y/Z + 7 digits + letter
static ES_DNI_REGEX: LazyLock<Regex> =
    LazyLock::new(|| regex(r"(?i)\b(?:\d{8}[A-Z]|[XYZ]\d{7}[A-Z])\b"));

// Dutch BSN (Burgerservicenummer): 9 digits
// Note: Not added to BUILTIN_PATTERNS - overlaps with ssn_nodash pattern
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "Public API - used by consumers")
)]
static NL_BSN_REGEX: LazyLock<Regex> = LazyLock::new(|| regex(r"\b\d{9}\b"));

// Generic European ID pattern (covers many formats)
// Matches patterns like: XX-123456, XX123456789, etc.
static EU_ID_REGEX: LazyLock<Regex> = LazyLock::new(|| regex(r"(?i)\b[A-Z]{1,3}[-\s]?\d{6,12}\b"));

// Driver's license patterns (various formats)
// Covers: LOUMA.657200.9.504, MASCU910077MV815, HERNA-607199-HK-599, etc.
static DRIVERS_LICENSE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    regex(
        r"(?xi)
        \b
        (?:
            # Format: NAME.DIGITS.X.DIGITS or NAME-DIGITS-XX-DIGITS
            [A-Z]{3,6}[.\-][0-9]{5,7}[.\-][A-Z0-9]{1,2}[.\-][0-9]{2,4} |
            # Format: NAMEYYYYMMDDXXXX (birth date embedded)
            [A-Z]{5}[0-9]{9,12}[A-Z]{0,3}[0-9]{0,3} |
            # Format: XNNN... (letter + digits, 9+ total)
            [A-Z][0-9]{8,12} |
            # Format: ALPHANUMERIC with embedded digits (10+ chars)
            [A-Z][0-9]{2,3}[A-Z0-9]{5,12} |
            # Explicit DL prefix
            (?:DL|D\.?L\.?|License|Lic)[:\-\s]?[A-Z0-9]{5,15}
        )
        \b
        ",
    )
});

// =============================================================================
// Pattern Registry
// =============================================================================

/// All built-in PII patterns.
pub static BUILTIN_PATTERNS: &[PiiPattern] = &[
    PiiPattern {
        name: "email",
        description: "Email address",
        regex: &EMAIL_REGEX,
        confidence: Confidence::High,
        category: PiiCategory::Contact,
        example: "user@example.com",
        replacement_template: "<EMAIL>",
    },
    PiiPattern {
        name: "phone_us",
        description: "US phone number",
        regex: &PHONE_US_REGEX,
        confidence: Confidence::High,
        category: PiiCategory::Contact,
        example: "(555) 234-5678",
        replacement_template: "<PHONE>",
    },
    PiiPattern {
        name: "phone_intl",
        description: "International phone number (E.164)",
        regex: &PHONE_INTL_REGEX,
        confidence: Confidence::High,
        category: PiiCategory::Contact,
        example: "+44123456789",
        replacement_template: "<PHONE>",
    },
    PiiPattern {
        name: "ssn",
        description: "US Social Security Number",
        regex: &SSN_REGEX,
        confidence: Confidence::High,
        category: PiiCategory::Identity,
        example: "123-45-6789",
        replacement_template: "<SSN>",
    },
    PiiPattern {
        name: "ssn_nodash",
        description: "US Social Security Number (no dashes)",
        regex: &SSN_NODASH_REGEX,
        confidence: Confidence::Medium,
        category: PiiCategory::Identity,
        example: "123456789",
        replacement_template: "<SSN>",
    },
    PiiPattern {
        name: "credit_card",
        description: "Credit card number",
        regex: &CREDIT_CARD_REGEX,
        confidence: Confidence::High,
        category: PiiCategory::Financial,
        example: "4111-1111-1111-1111",
        replacement_template: "<CREDIT_CARD>",
    },
    PiiPattern {
        name: "credit_card_nodash",
        description: "Credit card number (no separators)",
        regex: &CREDIT_CARD_NODASH_REGEX,
        confidence: Confidence::Medium,
        category: PiiCategory::Financial,
        example: "4111111111111111",
        replacement_template: "<CREDIT_CARD>",
    },
    PiiPattern {
        name: "ipv4",
        description: "IPv4 address",
        regex: &IPV4_REGEX,
        confidence: Confidence::High,
        category: PiiCategory::Network,
        example: "192.168.1.1",
        replacement_template: "<IPV4>",
    },
    PiiPattern {
        name: "ipv6",
        description: "IPv6 address",
        regex: &IPV6_REGEX,
        confidence: Confidence::High,
        category: PiiCategory::Network,
        example: "2001:0db8:85a3:0000:0000:8a2e:0370:7334",
        replacement_template: "<IPV6>",
    },
    PiiPattern {
        name: "mac",
        description: "MAC address",
        regex: &MAC_REGEX,
        confidence: Confidence::High,
        category: PiiCategory::Network,
        example: "00:1A:2B:3C:4D:5E",
        replacement_template: "<MAC>",
    },
    PiiPattern {
        name: "uuid",
        description: "UUID",
        regex: &UUID_REGEX,
        confidence: Confidence::High,
        category: PiiCategory::Other,
        example: "550e8400-e29b-41d4-a716-446655440000",
        replacement_template: "<UUID>",
    },
    PiiPattern {
        name: "jwt",
        description: "JSON Web Token",
        regex: &JWT_REGEX,
        confidence: Confidence::High,
        category: PiiCategory::Authentication,
        example: "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.abc123",
        replacement_template: "<JWT>",
    },
    PiiPattern {
        name: "aws_key",
        description: "AWS Access Key ID",
        regex: &AWS_KEY_REGEX,
        confidence: Confidence::High,
        category: PiiCategory::Authentication,
        example: "AKIAIOSFODNN7EXAMPLE",
        replacement_template: "<AWS_KEY>",
    },
    PiiPattern {
        name: "api_key",
        description: "Generic API key",
        regex: &API_KEY_REGEX,
        confidence: Confidence::Medium,
        category: PiiCategory::Authentication,
        example: "sk_live_abcdefghijklmnopqrstuvwxyz123456",
        replacement_template: "<API_KEY>",
    },
    PiiPattern {
        name: "vendor_secret",
        description: "Vendor-prefixed API token (Anthropic, OpenAI, GitHub, Slack, ...)",
        regex: &VENDOR_SECRET_REGEX,
        confidence: Confidence::High,
        category: PiiCategory::Authentication,
        example: "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        replacement_template: "<API_KEY>",
    },
    PiiPattern {
        name: "secret_assignment",
        description: "Secret value in an environment-style assignment",
        regex: &SECRET_ASSIGNMENT_REGEX,
        confidence: Confidence::High,
        category: PiiCategory::Authentication,
        example: "DB_PASSWORD=correct-horse-battery",
        replacement_template: "<SECRET>",
    },
    PiiPattern {
        name: "url_credentials",
        description: "Password embedded in a URL (scheme://user:password@host)",
        regex: &URL_CREDENTIALS_REGEX,
        confidence: Confidence::Medium,
        category: PiiCategory::Authentication,
        example: "postgres://app:Kq7vR2xLp9@db.internal:5432/orders",
        replacement_template: "<SECRET>",
    },
    PiiPattern {
        name: "private_key",
        description: "PEM private-key block",
        regex: &PRIVATE_KEY_REGEX,
        confidence: Confidence::High,
        category: PiiCategory::Authentication,
        example: "-----BEGIN PRIVATE KEY-----...-----END PRIVATE KEY-----",
        replacement_template: "<PRIVATE_KEY>",
    },
    PiiPattern {
        name: "iban",
        description: "International Bank Account Number",
        regex: &IBAN_REGEX,
        confidence: Confidence::High,
        category: PiiCategory::Financial,
        example: "DE89370400440532013000",
        replacement_template: "<IBAN>",
    },
    PiiPattern {
        name: "passport_us",
        description: "US Passport number",
        regex: &PASSPORT_US_REGEX,
        confidence: Confidence::Low,
        category: PiiCategory::Identity,
        example: "A12345678",
        replacement_template: "<PASSPORT>",
    },
    PiiPattern {
        name: "date",
        description: "Date (various formats)",
        regex: &DATE_REGEX,
        confidence: Confidence::Medium,
        category: PiiCategory::Other,
        example: "2024-12-25",
        replacement_template: "<DATE>",
    },
    PiiPattern {
        name: "time",
        description: "Time (HH:MM format)",
        regex: &TIME_REGEX,
        confidence: Confidence::Low,
        category: PiiCategory::Other,
        example: "14:30",
        replacement_template: "<TIME>",
    },
    PiiPattern {
        name: "username",
        description: "Username (user123 or explicitly labeled account)",
        regex: &USERNAME_REGEX,
        confidence: Confidence::Low,
        category: PiiCategory::Social,
        example: "user123",
        replacement_template: "<USERNAME>",
    },
    PiiPattern {
        name: "home_dir",
        description: "Home-directory path (/Users/user, /home/user)",
        regex: &HOME_DIR_REGEX,
        confidence: Confidence::High,
        category: PiiCategory::Social,
        example: "/Users/johndoe/project",
        replacement_template: "<HOME_DIR>",
    },
    PiiPattern {
        name: "unix_path",
        description: "Unix system path (/etc, /var, /opt, /usr)",
        regex: &UNIX_PATH_REGEX,
        confidence: Confidence::High,
        category: PiiCategory::Other,
        example: "/etc/nginx/nginx.conf",
        replacement_template: "<PATH>",
    },
    PiiPattern {
        name: "twitter_handle",
        description: "Twitter/X handle (@username, 1-15 chars)",
        regex: &TWITTER_HANDLE_REGEX,
        confidence: Confidence::High,
        category: PiiCategory::Social,
        example: "@jack",
        replacement_template: "<HANDLE>",
    },
    PiiPattern {
        name: "social_handle",
        description: "Social media handle (@username)",
        regex: &SOCIAL_HANDLE_REGEX,
        confidence: Confidence::Medium,
        category: PiiCategory::Social,
        example: "@johndoe",
        replacement_template: "<HANDLE>",
    },
    PiiPattern {
        name: "social_url",
        description: "Social media profile URL",
        regex: &SOCIAL_URL_REGEX,
        confidence: Confidence::High,
        category: PiiCategory::Social,
        example: "https://twitter.com/johndoe",
        replacement_template: "<SOCIAL_URL>",
    },
    PiiPattern {
        name: "geocoord",
        description: "GPS coordinates (latitude, longitude)",
        regex: &GEOCOORD_REGEX,
        confidence: Confidence::Medium,
        category: PiiCategory::Contact,
        example: "40.7128,-74.0060",
        replacement_template: "<GEOCOORD>",
    },
    PiiPattern {
        name: "uk_nino",
        description: "UK National Insurance Number",
        regex: &UK_NINO_REGEX,
        confidence: Confidence::High,
        category: PiiCategory::Identity,
        example: "AB123456C",
        replacement_template: "<UK_NINO>",
    },
    PiiPattern {
        name: "fr_nir",
        description: "French Social Security Number (NIR)",
        regex: &FR_NIR_REGEX,
        confidence: Confidence::High,
        category: PiiCategory::Identity,
        example: "1 85 12 75 108 108 42",
        replacement_template: "<FR_NIR>",
    },
    PiiPattern {
        name: "it_cf",
        description: "Italian Fiscal Code (Codice Fiscale)",
        regex: &IT_CF_REGEX,
        confidence: Confidence::High,
        category: PiiCategory::Identity,
        example: "RSSMRA85M01H501U",
        replacement_template: "<IT_CF>",
    },
    PiiPattern {
        name: "es_dni",
        description: "Spanish DNI/NIE",
        regex: &ES_DNI_REGEX,
        confidence: Confidence::High,
        category: PiiCategory::Identity,
        example: "12345678Z",
        replacement_template: "<ES_DNI>",
    },
    PiiPattern {
        name: "eu_id",
        description: "European ID number (generic)",
        regex: &EU_ID_REGEX,
        confidence: Confidence::Low,
        category: PiiCategory::Identity,
        example: "DE-123456789",
        replacement_template: "<EU_ID>",
    },
    PiiPattern {
        name: "drivers_license",
        description: "Driver's license number",
        regex: &DRIVERS_LICENSE_REGEX,
        confidence: Confidence::Medium,
        category: PiiCategory::Identity,
        example: "DL-A1234567",
        replacement_template: "<DRIVERS_LICENSE>",
    },
];

/// Get a pattern by name.
pub fn get_pattern(name: &str) -> Option<&'static PiiPattern> {
    BUILTIN_PATTERNS.iter().find(|p| p.name == name)
}

/// Get all pattern names.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "Public API - used by consumers")
)]
pub fn pattern_names() -> impl Iterator<Item = &'static str> {
    BUILTIN_PATTERNS.iter().map(|p| p.name)
}

/// Get patterns filtered by confidence level.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "Public API - used by consumers")
)]
pub fn patterns_by_confidence(min_confidence: Confidence) -> Vec<&'static PiiPattern> {
    BUILTIN_PATTERNS
        .iter()
        .filter(|p| p.confidence >= min_confidence)
        .collect()
}

/// Get patterns filtered by category.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "Public API - used by consumers")
)]
pub fn patterns_by_category(category: PiiCategory) -> Vec<&'static PiiPattern> {
    BUILTIN_PATTERNS
        .iter()
        .filter(|p| p.category == category)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        identifier_like, in_credential_field, non_address_email, non_secret_url_password,
        non_secret_value, strictly_inside_opaque,
    };

    #[test]
    fn opaque_runs_cover_ids_hashes_and_base64_but_not_numbers() {
        let uuid = "3f2a9c1e-7b4d-4e8a-9f1c-2d6b8e0a5c37";
        assert!(strictly_inside_opaque(uuid, 0, 8), "a uuid segment");
        assert!(
            !strictly_inside_opaque(uuid, 0, uuid.len()),
            "a whole id is a token"
        );
        let digest = "9a8e94ea7f378bd32cbbd37198a4a91436180c5bb472411e48b5ec2e2124ae9e";
        assert!(strictly_inside_opaque(digest, 5, 9));
        let base64 =
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQ";
        assert!(strictly_inside_opaque(base64, 10, 24));
        assert!(
            !strictly_inside_opaque("call 4111111111111111 now", 5, 21),
            "a card is not an id"
        );
        assert!(!strictly_inside_opaque("esi 4711 ok", 4, 8));
        // A name in a path or a sentence is not inside an id.
        assert!(!strictly_inside_opaque(
            "see /Users/quillon/app for it",
            14,
            21
        ));
        assert!(!strictly_inside_opaque("done. Quillon wrote", 6, 13));
    }

    #[test]
    fn credential_fields_are_recognised_by_the_key_before_the_value() {
        assert!(in_credential_field("api_key = 9a8e94ea7f378bd32cbb", 10));
        assert!(in_credential_field("REFRESH_TOKEN: abcdef0123456789", 15));
        assert!(!in_credential_field("let id = 9a8e94ea7f378bd32cbb", 9));
        assert!(!in_credential_field("hash 9a8e94ea7f378bd32cbb", 5));
        // A multi-byte character inside the look-behind window must not panic.
        let text = "Grüße aus München, ääää ääää; api_key = 9a8e94ea7f378bd32cbb";
        assert!(in_credential_field(text, text.find("9a8e").unwrap()));
        assert!(!in_credential_field("äöüß 4711", 8));
    }

    #[test]
    fn code_identifiers_are_not_people() {
        for value in [
            "mcp__linear__save_issue",
            "--dry-run",
            "std::io::Error",
            "src/agent/loop.ts",
            "THINKING_LEVEL",
            "a.b.c",
        ] {
            assert!(identifier_like(value), "{value} should read as code");
        }
        for value in [
            "Quillon",
            "Alice Example",
            "Müller",
            "firstChangedLine",
            "Claude Opus",
            "St.John",
            "qwen3",
            // Plain snake_case stays: a codename or username is indistinguishable.
            "max_retries",
            "nordwind_core",
        ] {
            assert!(!identifier_like(value), "{value} should read as a name");
        }
    }

    #[test]
    fn ssh_remotes_and_asset_names_are_not_addresses() {
        assert!(non_address_email("git@github.com"));
        assert!(non_address_email("logo@3x.png"));
        assert!(non_address_email("intro@2x.mp4"));
        assert!(!non_address_email("quillon.example@posteo.de"));
        assert!(!non_address_email("user7@example.invalid"));
    }

    #[test]
    fn references_masks_and_named_fixtures_are_not_secrets() {
        for value in [
            "env:OPENAI_API_KEY",
            "keychain:eavs-openai",
            "kyz:team/openai",
            "op://Private/openai/credential",
            "${DB_PASSWORD}",
            "{{ secrets.TOKEN }}",
            "<your-api-key>",
            "%API_TOKEN%",
            "process.env.API_KEY",
            "os.environ['TOKEN']",
            "xxxxxxxxxxxx",
            "********",
            "synthetic-access",
            "test-key-only",
            "your-token-here",
            "sk-local-xxx",
        ] {
            assert!(non_secret_value(value), "{value} should not be a secret");
        }
    }

    #[test]
    fn real_shapes_and_passphrases_stay_secrets() {
        for value in [
            "correct-horse-battery",
            "Sup3rSecretValue1",
            "SyntheticOnly-Pass-7zQ!",
            "sk_test_4eC39HqLyjWDarjtT1zdp7dc4eC39Hq",
            "test-Kq7vR2xLp9WmT4",
            "envoy-upstream-key",
        ] {
            assert!(!non_secret_value(value), "{value} should stay a secret");
        }
    }

    #[test]
    fn url_passwords_skip_defaults_and_references_only() {
        assert!(non_secret_url_password("postgres", "postgres"));
        assert!(non_secret_url_password("app", "password"));
        assert!(non_secret_url_password("app", "${DB_PASS}"));
        assert!(!non_secret_url_password("app", "Kq7vR2xLp9"));
        assert!(!non_secret_url_password("app", "hunter2pass"));
    }

    use super::*;

    #[test]
    fn test_email_pattern() {
        let re = &*EMAIL_REGEX;
        assert!(re.is_match("user@example.com"));
        assert!(re.is_match("first.last@sub.domain.org"));
        assert!(re.is_match("user+tag@example.co.uk"));
        assert!(!re.is_match("not-an-email"));
        assert!(!re.is_match("@missing-local.com"));
    }

    #[test]
    fn local_ipv4_ranges_are_not_personal() {
        for value in [
            "127.0.0.1",
            "10.1.2.3",
            "172.16.0.1",
            "172.31.255.254",
            "192.168.1.1",
            "169.254.10.1",
            "100.64.0.1",
            "100.127.255.254",
            "0.0.0.0",
            "224.0.0.1",
            "255.255.255.255",
            "192.0.2.5",
            "198.51.100.7",
            "203.0.113.9",
            "198.18.0.1",
        ] {
            assert!(
                is_local_ipv4(value),
                "{value} should be local/infrastructure"
            );
        }
        for value in [
            "8.8.8.8",
            "1.1.1.1",
            "100.63.255.255",
            "100.128.0.1",
            "172.32.0.1",
            "203.0.114.1",
        ] {
            assert!(!is_local_ipv4(value), "{value} should stay reportable");
        }
        assert!(!is_local_ipv4("not.an.ip"));
    }

    #[test]
    fn iban_spans_require_country_length_and_checksum() {
        fn span(text: &str) -> Option<&str> {
            IBAN_REGEX
                .find(text)
                .and_then(|m| iban_span(text, m.start(), m.end()))
                .map(|range| &text[range])
        }
        assert_eq!(
            span("IBAN: DE89370400440532013000"),
            Some("DE89370400440532013000")
        );
        assert_eq!(
            span("IBAN DE89 3704 0044 0532 0130 00 TEST"),
            Some("DE89 3704 0044 0532 0130 00")
        );
        assert_eq!(
            span("to GB82 WEST 1234 5698 7654 32."),
            Some("GB82 WEST 1234 5698 7654 32")
        );
        // Wrong check digits, wrong country length, and a truncated token.
        assert_eq!(span("DE88370400440532013000"), None);
        assert_eq!(span("DE8937040044053201300"), None);
        assert_eq!(span("DE89370400440532013000X1"), None);
    }

    #[test]
    fn test_phone_us_pattern() {
        let re = &*PHONE_US_REGEX;
        // Valid US phone numbers (exchange must start with 2-9)
        assert!(re.is_match("(555) 234-4567"));
        assert!(re.is_match("555-234-4567"));
        assert!(re.is_match("5552344567"));
        assert!(re.is_match("+1 555 234 4567"));
        assert!(!re.is_match("123-456")); // Too short
    }

    #[test]
    fn test_ssn_pattern() {
        let re = &*SSN_REGEX;
        assert!(re.is_match("123-45-6789"));
        assert!(!re.is_match("123456789")); // No dashes
        assert!(!re.is_match("12-345-6789")); // Wrong format
    }

    #[test]
    fn test_credit_card_pattern() {
        let re = &*CREDIT_CARD_REGEX;
        assert!(re.is_match("4111-1111-1111-1111")); // Visa
        assert!(re.is_match("5500 0000 0000 0004")); // Mastercard
        // Note: Amex has 15 digits with different grouping, tested separately
    }

    #[test]
    fn test_ipv4_pattern() {
        let re = &*IPV4_REGEX;
        assert!(re.is_match("192.168.1.1"));
        assert!(re.is_match("10.0.0.1"));
        assert!(re.is_match("255.255.255.255"));
        assert!(!re.is_match("256.1.1.1")); // Invalid octet
        assert!(!re.is_match("1.2.3")); // Missing octet
    }

    #[test]
    fn test_uuid_pattern() {
        let re = &*UUID_REGEX;
        assert!(re.is_match("550e8400-e29b-41d4-a716-446655440000"));
        assert!(re.is_match("550E8400-E29B-41D4-A716-446655440000")); // Uppercase
        assert!(!re.is_match("550e8400-e29b-41d4-a716")); // Too short
    }

    #[test]
    fn test_jwt_pattern() {
        let re = &*JWT_REGEX;
        assert!(re.is_match(
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.TJVA95OrM7E2cBab30RMHrHDcEfxjoYZgeFONFh7HgQ"
        ));
        assert!(!re.is_match("not.a.jwt"));
    }

    #[test]
    fn test_home_dir_pattern() {
        let re = &*HOME_DIR_REGEX;
        assert!(re.is_match("/Users/tommyfalkowski/project"));
        assert!(re.is_match("/home/wismut/.config"));
        assert!(re.is_match("/Volumes/Macintosh HD/apps"));
        // Unicode usernames redact in full, not a partial ASCII prefix.
        assert!(re.is_match("/Users/müller/apps"));
        assert!(re.is_match("/home/東京/projects"));
        assert!(re.is_match("/Users/joão/site"));
        // Windows home dirs, both separators.
        assert!(re.is_match(r"C:\Users\Jörg\Desktop"));
        assert!(re.is_match("C:/Users/john/AppData"));
        assert!(!re.is_match("/usr/local/bin/python")); // not a home path
        assert!(!re.is_match("homepage"));
    }

    #[test]
    fn test_unix_path_pattern() {
        let re = &*UNIX_PATH_REGEX;
        assert!(re.is_match("/etc/nginx/nginx.conf"));
        assert!(re.is_match("/var/log/syslog"));
        assert!(re.is_match("/opt/homebrew/bin/uv"));
        assert!(re.is_match("/usr/local/bin/python"));
        assert!(re.is_match("/root/.ssh/authorized_keys"));
        // Non-ASCII host/config names captured whole.
        assert!(re.is_match("/var/www/münchen/index.html"));
        assert!(!re.is_match("vim"));
    }

    #[test]
    fn test_username_code_false_positives() {
        use crate::engine::detector::{Detector, DetectorConfig};

        let detector = Detector::new(
            &DetectorConfig::default()
                .with_patterns(["username"])
                .with_min_confidence(Confidence::Low),
        )
        .unwrap();
        let text = "line1 line2 line3 line4 api_path GRAFANA_API_KEY \
                    550e8400-e29b-41d4-a716-446655440000 timeoutMs";
        assert!(detector.detect(text).unwrap().is_empty());
    }

    #[test]
    fn test_username_explicit_accounts() {
        let re = &*USERNAME_REGEX;
        for text in [
            "user123",
            "user7",
            "username=john_doe",
            "login: alice42",
            "\"username\":\"john_doe\"",
        ] {
            assert_eq!(re.find(text).map(|m| m.as_str()), Some(text));
        }
    }

    #[test]
    fn test_check_pattern() {
        let re = &*SOCIAL_HANDLE_REGEX;
        assert!(!re.is_match("none"));
    }

    #[test]
    fn test_get_pattern() {
        assert!(get_pattern("email").is_some());
        assert!(get_pattern("nonexistent").is_none());
    }

    #[test]
    fn test_patterns_by_confidence() {
        let high = patterns_by_confidence(Confidence::High);
        assert!(high.iter().all(|p| p.confidence == Confidence::High));

        let medium = patterns_by_confidence(Confidence::Medium);
        assert!(medium.len() >= high.len());
    }

    #[test]
    fn test_social_handle_pattern() {
        let re = &*SOCIAL_HANDLE_REGEX;
        assert!(re.is_match("@johndoe"));
        assert!(re.is_match("@John_Doe123"));
        assert!(re.is_match("@user.name"));
        assert!(re.is_match("Follow me @username here"));
        // Note: The regex itself matches @domain.com - filtering is done in detector
        assert!(!re.is_match("@")); // Just @ sign
        assert!(!re.is_match("@1starting_with_number")); // Must start with letter
    }

    #[test]
    fn test_twitter_handle_pattern() {
        let re = &*TWITTER_HANDLE_REGEX;
        assert!(re.is_match("@jack"));
        assert!(re.is_match("@elonmusk"));
        assert!(re.is_match("@user_name"));
        assert!(re.is_match("@_underscore"));
        // Twitter handles are max 15 chars
        assert!(!re.is_match("@thisusernameiswaytoolong"));
    }

    #[test]
    fn test_social_url_pattern() {
        let re = &*SOCIAL_URL_REGEX;
        assert!(re.is_match("https://twitter.com/johndoe"));
        assert!(re.is_match("https://x.com/elonmusk"));
        assert!(re.is_match("https://github.com/torvalds"));
        assert!(re.is_match("https://instagram.com/nasa"));
        assert!(!re.is_match("https://example.com/page"));
    }

    #[test]
    fn test_social_handle_not_in_email() {
        // This tests the detector-level filtering, not the regex
        use crate::engine::detector::{Detector, DetectorConfig};

        let config = DetectorConfig::default()
            .with_patterns(["email", "social_handle"])
            .with_min_confidence(Confidence::Medium);
        let detector = Detector::new(&config).unwrap();

        // Email should be detected, but @domain should NOT be detected as a handle
        let matches = detector.detect("Contact test@example.com").unwrap();

        let pattern_names: Vec<&str> = matches.iter().map(|m| m.pattern_name.as_str()).collect();
        assert!(pattern_names.contains(&"email"), "Should detect email");
        assert!(
            !pattern_names.contains(&"social_handle"),
            "Should NOT detect @example as social handle"
        );
    }
}
