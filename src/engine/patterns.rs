//! PII detection patterns.
//!
//! This module defines all built-in regex patterns for detecting
//! personally identifiable information (PII).

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};

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
    pub regex: &'static Lazy<Regex>,
    /// Confidence level
    pub confidence: Confidence,
    /// Category of PII
    pub category: PiiCategory,
    /// Example of what this pattern matches
    pub example: &'static str,
    /// Default replacement placeholder for this pattern type.
    /// Used by the placeholder replacement strategy.
    #[allow(dead_code)]
    pub replacement_template: &'static str,
}

// =============================================================================
// Pattern Definitions
// =============================================================================

// Email pattern
static EMAIL_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b").unwrap());

// US Phone number patterns
static PHONE_US_REGEX: Lazy<Regex> = Lazy::new(|| {
    // Matches: (555) 123-4567, 555-123-4567, 555.123.4567, 5551234567, +1 555 123 4567
    Regex::new(r"(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?[2-9]\d{2}[-.\s]?\d{4}").unwrap()
});

// International phone (E.164 format, with optional spaces/separators)
static PHONE_INTL_REGEX: Lazy<Regex> = Lazy::new(|| {
    // Matches: +49192836418, +49 192836418, +49 192 836 418, +49-192-836-418
    Regex::new(r"\+[1-9][\d\s\-]{6,17}\d\b").unwrap()
});

// US Social Security Number (with dashes, dots, or spaces)
static SSN_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b\d{3}[-.\s]\d{2,3}[-.\s]\d{4}\b").unwrap()
});

// SSN without dashes (9 consecutive digits with word boundary)
static SSN_NODASH_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b\d{9}\b").unwrap());

// Credit card numbers (major brands)
static CREDIT_CARD_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
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
    .unwrap()
});

// Credit card without separators
static CREDIT_CARD_NODASH_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(?:4\d{15}|5[1-5]\d{14}|3[47]\d{13}|6(?:011|5\d{2})\d{12})\b").unwrap()
});

// IPv4 address
static IPV4_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        \b
        (?:
            (?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.
        ){3}
        (?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)
        \b
        ",
    )
    .unwrap()
});

// IPv6 address (simplified - matches common formats)
static IPV6_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\b(?:[a-f0-9]{1,4}:){7}[a-f0-9]{1,4}\b").unwrap());

// MAC address
static MAC_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\b(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}\b").unwrap());

// UUID
static UUID_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b").unwrap()
});

// JWT token
static JWT_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*").unwrap());

// AWS Access Key ID
static AWS_KEY_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?:A3T[A-Z0-9]|AKIA|AGPA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}").unwrap()
});

// Generic API key pattern (40+ char alphanumeric)
static API_KEY_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(?:sk|pk|api|key|token)[-_]?(?:live|test|prod)?[-_]?[a-zA-Z0-9]{32,}\b").unwrap()
});

// IBAN (International Bank Account Number)
static IBAN_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b").unwrap());

// US Passport
static PASSPORT_US_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b[A-Z]\d{8}\b").unwrap());

// Date patterns (various formats)
// ISO: 2024-12-25, US: 12/25/2024, EU: 25.12.2024, etc.
static DATE_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
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
    .unwrap()
});

// Time patterns: 14:30, 2:30 PM, 10:20am, 7 o'clock, quarter past 13, etc.
static TIME_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
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
    .unwrap()
});

// Username (without @, for matching dataset usernames)
static USERNAME_REGEX: Lazy<Regex> = Lazy::new(|| {
    // Common username patterns: user123, john_doe, etc.
    // Only match if it looks like a username (has numbers or underscores, or is in a context)
    Regex::new(r"\b[a-zA-Z][a-zA-Z0-9_]{2,20}\d+[a-zA-Z0-9_]*\b|\b[a-zA-Z][a-zA-Z0-9]*_[a-zA-Z0-9_]+\b").unwrap()
});

// Social media handles
// Twitter/X handle: @username (1-15 chars, alphanumeric + underscore)
// Note: We exclude emails by requiring @ to NOT be preceded by alphanumeric
// This is handled in post-processing since regex crate doesn't support look-behind
static TWITTER_HANDLE_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"@[a-zA-Z_][a-zA-Z0-9_]{0,14}\b").unwrap());

// Generic social handle (covers most platforms): @username
// More permissive than platform-specific patterns
static SOCIAL_HANDLE_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"@[a-zA-Z][a-zA-Z0-9_.]{1,30}\b").unwrap());

// URL with username path (e.g., twitter.com/username, github.com/username)
static SOCIAL_URL_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?i)https?://(?:www\.)?(?:twitter|x|instagram|facebook|fb|linkedin|github|tiktok|youtube|reddit)\.com/@?[a-zA-Z0-9_.-]{1,39}(?:\?|/|$)",
    )
    .unwrap()
});

// GPS coordinates (latitude, longitude)
// Must have decimal points to differentiate from other number pairs
// Matches: 40.7128,-74.0060, 40.7128, -74.0060, (40.7128, -74.0060)
static GEOCOORD_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        \(?
        [-+]?(?:[1-8]?\d\.\d{2,}|90\.0+)  # Latitude: -90 to 90, MUST have decimal with 2+ digits
        \s*[,]\s*                          # Comma separator (required for specificity)
        [-+]?(?:1[0-7]\d\.\d{2,}|180\.0+|\d{1,2}\.\d{2,})  # Longitude: -180 to 180
        \)?
        ",
    )
    .unwrap()
});

// European Social/Tax ID patterns
// German Tax ID (Steueridentifikationsnummer): 11 digits
// Note: Not added to BUILTIN_PATTERNS due to high false positive rate with generic 11-digit numbers
#[allow(dead_code)]
static DE_TAX_ID_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b\d{11}\b").unwrap()
});

// UK National Insurance Number: 2 letters, 6 digits, 1 letter
static UK_NINO_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b").unwrap()
});

// French Social Security Number (NIR): 13 digits + 2 digit key
static FR_NIR_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b").unwrap()
});

// Italian Fiscal Code (Codice Fiscale): 16 alphanumeric chars
static IT_CF_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b").unwrap()
});

// Spanish DNI/NIE: 8 digits + letter or X/Y/Z + 7 digits + letter
static ES_DNI_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:\d{8}[A-Z]|[XYZ]\d{7}[A-Z])\b").unwrap()
});

// Dutch BSN (Burgerservicenummer): 9 digits
// Note: Not added to BUILTIN_PATTERNS - overlaps with ssn_nodash pattern
#[allow(dead_code)]
static NL_BSN_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b\d{9}\b").unwrap()
});

// Generic European ID pattern (covers many formats)
// Matches patterns like: XX-123456, XX123456789, etc.
static EU_ID_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b[A-Z]{1,3}[-\s]?\d{6,12}\b").unwrap()
});

// Driver's license patterns (various formats)
// Covers: LOUMA.657200.9.504, MASCU910077MV815, HERNA-607199-HK-599, etc.
static DRIVERS_LICENSE_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
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
    .unwrap()
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
        name: "iban",
        description: "International Bank Account Number",
        regex: &IBAN_REGEX,
        confidence: Confidence::Medium,
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
        description: "Username (user123, john_doe style)",
        regex: &USERNAME_REGEX,
        confidence: Confidence::Low,
        category: PiiCategory::Social,
        example: "user123",
        replacement_template: "<USERNAME>",
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
#[allow(dead_code)]
pub fn pattern_names() -> impl Iterator<Item = &'static str> {
    BUILTIN_PATTERNS.iter().map(|p| p.name)
}

/// Get patterns filtered by confidence level.
#[allow(dead_code)]
pub fn patterns_by_confidence(min_confidence: Confidence) -> Vec<&'static PiiPattern> {
    BUILTIN_PATTERNS
        .iter()
        .filter(|p| match (min_confidence, p.confidence) {
            (Confidence::Low, _) => true,
            (Confidence::Medium, Confidence::Low) => false,
            (Confidence::Medium, _) => true,
            (Confidence::High, Confidence::High) => true,
            (Confidence::High, _) => false,
        })
        .collect()
}

/// Get patterns filtered by category.
#[allow(dead_code)]
pub fn patterns_by_category(category: PiiCategory) -> Vec<&'static PiiPattern> {
    BUILTIN_PATTERNS
        .iter()
        .filter(|p| p.category == category)
        .collect()
}

#[cfg(test)]
mod tests {
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
        let detector = Detector::new(&config);

        // Email should be detected, but @domain should NOT be detected as a handle
        let matches = detector.detect("Contact test@example.com");

        let pattern_names: Vec<&str> = matches.iter().map(|m| m.pattern_name.as_str()).collect();
        assert!(pattern_names.contains(&"email"), "Should detect email");
        assert!(
            !pattern_names.contains(&"social_handle"),
            "Should NOT detect @example as social handle"
        );
    }
}
