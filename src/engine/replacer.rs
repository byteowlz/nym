//! PII replacement strategies.
//!
//! This module provides different strategies for replacing detected PII.

use std::collections::HashMap;

use base64::Engine;
use fake::Fake;
use fake::faker::address::en::{CityName, StateAbbr, StreetName, ZipCode};
use fake::faker::company::en::CompanyName;
use fake::faker::internet::en::{IPv4, IPv6, MACAddress, Username};
use fake::faker::name::en::{FirstName, LastName, Name};
use fake::rand::rngs::StdRng;
use fake::rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::detector::PiiMatch;

/// Replacement strategy for PII.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ReplacementStrategy {
    /// Use a fixed placeholder (e.g., <EMAIL>)
    Placeholder,
    /// Mask with X characters, preserving length
    Mask,
    /// Generate a consistent hash-based replacement
    Hash,
    /// Generate random but realistic-looking replacements (legacy)
    Random,
    /// Use consistent replacements (same input = same output within session)
    Consistent,
    /// Generate realistic fake data (names, emails, addresses, etc.)
    #[default]
    Fake,
}

/// A component mapping for partial matches (e.g., first name, last name).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentMapping {
    /// Original component (e.g., "John")
    pub original: String,
    /// Replacement component (e.g., "Donald")
    pub replacement: String,
    /// Component type (e.g., "first_name", "last_name")
    pub component_type: String,
}

/// A replacement mapping entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Replacement {
    /// Original PII value
    pub original: String,
    /// Replacement value
    pub replacement: String,
    /// Pattern type that matched
    pub pattern_name: String,
    /// Component mappings for partial deanonymization (e.g., first/last name)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub components: Vec<ComponentMapping>,
}

/// Configuration for the replacer.
#[derive(Debug, Clone)]
pub struct ReplacerConfig {
    /// Replacement strategy
    pub strategy: ReplacementStrategy,
    /// Seed for deterministic random generation
    pub seed: Option<u64>,
    /// Domain to use for email replacements
    pub email_domain: String,
}

impl Default for ReplacerConfig {
    fn default() -> Self {
        Self {
            strategy: ReplacementStrategy::Placeholder,
            seed: None,
            email_domain: "example.com".to_string(),
        }
    }
}

impl ReplacerConfig {
    /// Create a config with placeholder strategy.
    #[allow(dead_code)]
    pub fn placeholder() -> Self {
        Self {
            strategy: ReplacementStrategy::Placeholder,
            ..Default::default()
        }
    }

    /// Create a config with mask strategy.
    #[allow(dead_code)]
    pub fn mask() -> Self {
        Self {
            strategy: ReplacementStrategy::Mask,
            ..Default::default()
        }
    }

    /// Create a config with hash strategy.
    #[allow(dead_code)]
    pub fn hash() -> Self {
        Self {
            strategy: ReplacementStrategy::Hash,
            ..Default::default()
        }
    }

    /// Create a config with consistent strategy.
    #[allow(dead_code)]
    pub fn consistent() -> Self {
        Self {
            strategy: ReplacementStrategy::Consistent,
            ..Default::default()
        }
    }

    /// Set a seed for deterministic output.
    #[allow(dead_code)]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the email domain for replacements.
    #[allow(dead_code)]
    pub fn with_email_domain(mut self, domain: impl Into<String>) -> Self {
        self.email_domain = domain.into();
        self
    }
}

/// Cached replacement with its components for consistency.
#[derive(Debug, Clone)]
struct CachedReplacement {
    /// The full replacement string
    replacement: String,
    /// Component mappings (for partial deanonymization)
    components: Vec<ComponentMapping>,
}

/// The PII replacer.
pub struct Replacer {
    config: ReplacerConfig,
    /// Cache for consistent replacements (used by Consistent and Fake strategies)
    /// Key is the normalized original text
    replacement_cache: HashMap<String, CachedReplacement>,
    /// Cache for name component mappings (normalized original → normalized replacement)
    /// Used for linking derived PII (e.g., email local parts containing names)
    component_cache: HashMap<String, String>,
    /// Counter for generating unique placeholders
    counter: usize,
    /// RNG for random replacements (uses fake's rand for compatibility)
    rng: StdRng,
    /// Session ID for embedding in replacements
    session_id: Option<String>,
}

impl Replacer {
    /// Create a new replacer with the given configuration.
    pub fn new(config: ReplacerConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        Self {
            config,
            replacement_cache: HashMap::new(),
            component_cache: HashMap::new(),
            counter: 0,
            rng,
            session_id: None,
        }
    }

    /// Create a replacer with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ReplacerConfig::default())
    }

    /// Set the session ID for embedding in replacements.
    pub fn with_session_id(mut self, session_id: String) -> Self {
        self.session_id = Some(session_id);
        self
    }

    /// Generate a replacement for a PII match.
    pub fn replace(&mut self, pii_match: &PiiMatch) -> Replacement {
        let (replacement, components) = match self.config.strategy {
            ReplacementStrategy::Placeholder => (self.placeholder_replacement(pii_match), vec![]),
            ReplacementStrategy::Mask => (self.mask_replacement(pii_match), vec![]),
            ReplacementStrategy::Hash => (self.hash_replacement(pii_match), vec![]),
            ReplacementStrategy::Random => (self.random_replacement(pii_match), vec![]),
            ReplacementStrategy::Consistent => self.consistent_replacement(pii_match),
            ReplacementStrategy::Fake => self.fake_replacement_with_components(pii_match),
        };

        Replacement {
            original: pii_match.matched_text.clone(),
            replacement,
            pattern_name: pii_match.pattern_name.clone(),
            components,
        }
    }

    /// Apply replacements to text, returning the modified text and all replacements.
    pub fn replace_all(&mut self, text: &str, matches: &[PiiMatch]) -> (String, Vec<Replacement>) {
        if matches.is_empty() {
            return (text.to_string(), Vec::new());
        }

        let mut result = String::with_capacity(text.len());
        let mut replacements = Vec::with_capacity(matches.len());
        let mut last_end = 0;

        // Process matches in order (they should already be sorted by start position)
        for pii_match in matches {
            // Skip overlapping matches
            if pii_match.start < last_end {
                continue;
            }

            // Add text before this match
            result.push_str(&text[last_end..pii_match.start]);

            // Generate and apply replacement
            let replacement = self.replace(pii_match);
            result.push_str(&replacement.replacement);
            replacements.push(replacement);

            last_end = pii_match.end;
        }

        // Add remaining text
        result.push_str(&text[last_end..]);

        (result, replacements)
    }

    /// Get the current configuration.
    #[allow(dead_code)]
    pub fn config(&self) -> &ReplacerConfig {
        &self.config
    }

    // -------------------------------------------------------------------------
    // Private replacement methods
    // -------------------------------------------------------------------------

    fn placeholder_replacement(&self, pii_match: &PiiMatch) -> String {
        // Use pattern-specific placeholders
        match pii_match.pattern_name.as_str() {
            "email" => "<EMAIL>".to_string(),
            "phone_us" | "phone_intl" => "<PHONE>".to_string(),
            "ssn" | "ssn_nodash" => "<SSN>".to_string(),
            "credit_card" | "credit_card_nodash" => "<CREDIT_CARD>".to_string(),
            "ipv4" => "<IPV4>".to_string(),
            "ipv6" => "<IPV6>".to_string(),
            "mac" => "<MAC>".to_string(),
            "uuid" => "<UUID>".to_string(),
            "jwt" => "<JWT>".to_string(),
            "aws_key" | "api_key" => "<API_KEY>".to_string(),
            "iban" => "<IBAN>".to_string(),
            "passport_us" => "<PASSPORT>".to_string(),
            "social_handle" | "twitter_handle" | "instagram_handle" => "<HANDLE>".to_string(),
            "social_url" => "<SOCIAL_URL>".to_string(),
            "date" => "<DATE>".to_string(),
            "time" => "<TIME>".to_string(),
            "username" => "<USERNAME>".to_string(),
            _ => "<PII>".to_string(),
        }
    }

    fn mask_replacement(&self, pii_match: &PiiMatch) -> String {
        // Preserve structure where possible
        pii_match
            .matched_text
            .chars()
            .map(|c| {
                if c.is_alphanumeric() {
                    'X'
                } else {
                    c // Preserve separators like -, ., @
                }
            })
            .collect()
    }

    fn hash_replacement(&mut self, pii_match: &PiiMatch) -> String {
        let mut hasher = Sha256::new();
        hasher.update(pii_match.matched_text.as_bytes());
        let hash = hasher.finalize();
        let hash_str = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&hash[..8]);

        // Format based on pattern type
        match pii_match.pattern_name.as_str() {
            "email" => format!("user_{}@{}", &hash_str[..6], self.config.email_domain),
            "phone_us" => format!("555-{}-{}", &hash_str[..3], &hash_str[3..7]),
            "ssn" => format!("XXX-XX-{}", &hash_str[..4]),
            _ => format!("<{}>", &hash_str[..8]),
        }
    }

    fn random_replacement(&mut self, pii_match: &PiiMatch) -> String {
        self.counter += 1;

        match pii_match.pattern_name.as_str() {
            "email" => {
                let user: u32 = self.rng.random_range(1000..9999);
                let domain = &self.config.email_domain;
                if let Some(ref session) = self.session_id {
                    format!("user{user}.{session}@{domain}")
                } else {
                    format!("user{user}@{domain}")
                }
            }
            "phone_us" => {
                let exchange: u32 = self.rng.random_range(200..999);
                let subscriber: u32 = self.rng.random_range(1000..9999);
                if let Some(ref session) = self.session_id {
                    format!("555-{exchange}-{subscriber}-{session}")
                } else {
                    format!("555-{exchange}-{subscriber}")
                }
            }
            "ssn" => {
                let last4: u32 = self.rng.random_range(1000..9999);
                if let Some(ref session) = self.session_id {
                    format!("XXX-XX-{last4}-{session}")
                } else {
                    format!("XXX-XX-{last4}")
                }
            }
            "credit_card" | "credit_card_nodash" => {
                if let Some(ref session) = self.session_id {
                    format!("XXXX-XXXX-XXXX-XXXX-{session}")
                } else {
                    "XXXX-XXXX-XXXX-XXXX".to_string()
                }
            }
            "ipv4" => {
                let a = self.rng.random_range(0..255);
                let b = self.rng.random_range(0..255);
                let c = self.rng.random_range(1..255);
                if let Some(ref session) = self.session_id {
                    format!("10.{a}.{b}.{c}.{session}")
                } else {
                    format!("10.{a}.{b}.{c}")
                }
            }
            "uuid" => {
                let uuid = format!(
                    "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
                    self.rng.random::<u32>(),
                    self.rng.random::<u16>(),
                    self.rng.random::<u16>(),
                    self.rng.random::<u16>(),
                    self.rng.random::<u64>() & 0xFFFFFFFFFFFF
                );
                if let Some(ref session) = self.session_id {
                    format!("{uuid}-{session}")
                } else {
                    uuid
                }
            }
            _ => {
                if let Some(ref session) = self.session_id {
                    format!("<REDACTED_{}-{}>", self.counter, session)
                } else {
                    format!("<REDACTED_{}>", self.counter)
                }
            }
        }
    }

    fn consistent_replacement(&mut self, pii_match: &PiiMatch) -> (String, Vec<ComponentMapping>) {
        // Normalize text for cache lookup (lowercase, trimmed)
        let cache_key = pii_match.matched_text.to_lowercase();

        // Check cache first
        if let Some(cached) = self.replacement_cache.get(&cache_key) {
            return (cached.replacement.clone(), cached.components.clone());
        }

        // Generate new replacement using fake data (not random placeholders)
        let (replacement, components) = self.generate_fake_replacement(pii_match);

        // Cache it
        self.replacement_cache.insert(
            cache_key,
            CachedReplacement {
                replacement: replacement.clone(),
                components: components.clone(),
            },
        );

        (replacement, components)
    }

    fn fake_replacement(&mut self, pii_match: &PiiMatch) -> String {
        self.counter += 1;

        match pii_match.pattern_name.as_str() {
            "email" => {
                // Generate realistic email with session ID embedded
                let username: String = Username().fake_with_rng(&mut self.rng);
                let domain = &self.config.email_domain;
                if let Some(ref session) = self.session_id {
                    format!("{username}.{session}@{domain}")
                } else {
                    format!("{username}@{domain}")
                }
            }
            "phone_us" | "phone_ner" => {
                // Generate US phone with 555 prefix (reserved for fiction)
                let exchange: u32 = self.rng.random_range(200..999);
                let subscriber: u32 = self.rng.random_range(1000..9999);
                format!("(555) {exchange}-{subscriber}")
            }
            "phone_intl" => {
                // Preserve country code from original, generate fake local number
                let original = &pii_match.matched_text;

                // Extract country code (e.g., +49, +44, +1)
                let country_code = if original.starts_with('+') {
                    // Find where digits end for country code (1-3 digits after +)
                    let code_end = original[1..]
                        .char_indices()
                        .take_while(|(i, c)| c.is_ascii_digit() && *i < 3)
                        .last()
                        .map(|(i, _)| i + 2)
                        .unwrap_or(1);
                    &original[..code_end]
                } else {
                    "+1" // Default to US if no country code
                };

                // Generate fake local number based on country
                match country_code {
                    "+49" => {
                        // German format: +49 XXX XXXXXXX (mobile) or +49 XX XXXXXXXX (landline)
                        let prefix: u32 = self.rng.random_range(150..179); // German mobile prefixes
                        let number: u32 = self.rng.random_range(1000000..9999999);
                        format!("+49 {prefix} {number}")
                    }
                    "+44" => {
                        // UK format: +44 XXXX XXXXXX
                        let area: u32 = self.rng.random_range(1000..9999);
                        let number: u32 = self.rng.random_range(100000..999999);
                        format!("+44 {area} {number}")
                    }
                    "+33" => {
                        // French format: +33 X XX XX XX XX
                        let prefix: u32 = self.rng.random_range(1..9);
                        let p1: u32 = self.rng.random_range(10..99);
                        let p2: u32 = self.rng.random_range(10..99);
                        let p3: u32 = self.rng.random_range(10..99);
                        let p4: u32 = self.rng.random_range(10..99);
                        format!("+33 {prefix} {p1:02} {p2:02} {p3:02} {p4:02}")
                    }
                    "+1" => {
                        // US/Canada format: +1 555-XXX-XXXX (555 is reserved for fiction)
                        let exchange: u32 = self.rng.random_range(200..999);
                        let subscriber: u32 = self.rng.random_range(1000..9999);
                        format!("+1 555-{exchange}-{subscriber}")
                    }
                    "+43" => {
                        // Austrian format: +43 XXX XXXXXXX
                        let prefix: u32 = self.rng.random_range(600..699);
                        let number: u32 = self.rng.random_range(1000000..9999999);
                        format!("+43 {prefix} {number}")
                    }
                    "+41" => {
                        // Swiss format: +41 XX XXX XX XX
                        let prefix: u32 = self.rng.random_range(70..79);
                        let p1: u32 = self.rng.random_range(100..999);
                        let p2: u32 = self.rng.random_range(10..99);
                        let p3: u32 = self.rng.random_range(10..99);
                        format!("+41 {prefix} {p1} {p2} {p3}")
                    }
                    _ => {
                        // Generic international format: preserve country code + random digits
                        let number: u64 = self.rng.random_range(100000000..999999999);
                        format!("{country_code} {number}")
                    }
                }
            }
            "ssn" | "ssn_nodash" => {
                // Generate fake SSN (using reserved ranges)
                let area: u32 = self.rng.random_range(900..999); // Reserved range
                let group: u32 = self.rng.random_range(10..99);
                let serial: u32 = self.rng.random_range(1000..9999);
                if pii_match.pattern_name == "ssn_nodash" {
                    format!("{area}{group:02}{serial}")
                } else {
                    format!("{area}-{group:02}-{serial}")
                }
            }
            "credit_card" | "credit_card_nodash" => {
                // Generate fake credit card (test card format)
                let nums: Vec<u32> = (0..12).map(|_| self.rng.random_range(0..10)).collect();
                let card = format!(
                    "4111{}{}{}{}{}{}{}{}{}{}{}{}",
                    nums[0],
                    nums[1],
                    nums[2],
                    nums[3],
                    nums[4],
                    nums[5],
                    nums[6],
                    nums[7],
                    nums[8],
                    nums[9],
                    nums[10],
                    nums[11]
                );
                if pii_match.pattern_name == "credit_card_nodash" {
                    card
                } else {
                    format!(
                        "{}-{}-{}-{}",
                        &card[0..4],
                        &card[4..8],
                        &card[8..12],
                        &card[12..16]
                    )
                }
            }
            "ipv4" => {
                let ip: String = IPv4().fake_with_rng(&mut self.rng);
                // Use private range (10.x.x.x)
                format!("10.{}", &ip[ip.find('.').map(|i| i + 1).unwrap_or(0)..])
            }
            "ipv6" => {
                let ip: String = IPv6().fake_with_rng(&mut self.rng);
                ip
            }
            "mac" => {
                let mac: String = MACAddress().fake_with_rng(&mut self.rng);
                mac
            }
            "uuid" => {
                // Generate a random UUID
                let uuid = format!(
                    "{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
                    self.rng.random::<u32>(),
                    self.rng.random::<u16>(),
                    self.rng.random::<u16>() & 0x0FFF,
                    (self.rng.random::<u16>() & 0x3FFF) | 0x8000,
                    self.rng.random::<u64>() & 0xFFFFFFFFFFFF
                );
                uuid
            }
            "jwt" => {
                // Generate fake JWT structure
                let header = base64::engine::general_purpose::URL_SAFE_NO_PAD
                    .encode(r#"{"alg":"HS256","typ":"JWT"}"#);
                let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(format!(
                    r#"{{"sub":"{}","iat":{}}}"#,
                    self.counter,
                    chrono::Utc::now().timestamp()
                ));
                let sig: [u8; 32] = self.rng.random();
                let signature = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(sig);
                format!("{header}.{payload}.{signature}")
            }
            "aws_key" | "api_key" => {
                // Generate fake API key
                let key: [u8; 20] = self.rng.random();
                let encoded = base64::engine::general_purpose::STANDARD.encode(key);
                if pii_match.pattern_name == "aws_key" {
                    format!("AKIA{}", &encoded[..16].to_uppercase())
                } else {
                    encoded
                }
            }
            "iban" => {
                // Generate fake IBAN (using test country code)
                let checksum: u32 = self.rng.random_range(10..99);
                let bban: u64 = self.rng.random::<u64>() % 10_000_000_000_000_000;
                format!("XX{checksum}{bban:016}")
            }
            "passport_us" => {
                // Generate fake passport number
                let num: u32 = self.rng.random_range(10000000..99999999);
                format!("X{num}")
            }
            "social_handle" | "twitter_handle" | "instagram_handle" => {
                // Generate fake social media handle
                // Try to link to known name components first
                let original = &pii_match.matched_text;
                let handle_part = original.trim_start_matches('@');

                // Check if handle contains known name parts
                let (replaced, was_linked) = self.replace_with_components(handle_part);

                if was_linked {
                    // Use linked name parts
                    let normalized = replaced
                        .to_lowercase()
                        .replace(' ', "_")
                        .chars()
                        .filter(|c| c.is_alphanumeric() || *c == '_')
                        .collect::<String>();
                    format!("@{normalized}")
                } else {
                    // Generate random username
                    let username: String = Username().fake_with_rng(&mut self.rng);
                    // Make it look more like a social handle
                    let num: u32 = self.rng.random_range(0..999);
                    format!("@{username}{num}")
                }
            }
            "social_url" => {
                // Generate fake social media URL
                let original = &pii_match.matched_text;

                // Extract platform from URL
                let platform = if original.contains("twitter.com") || original.contains("x.com") {
                    "twitter.com"
                } else if original.contains("instagram.com") {
                    "instagram.com"
                } else if original.contains("github.com") {
                    "github.com"
                } else if original.contains("linkedin.com") {
                    "linkedin.com"
                } else if original.contains("facebook.com") || original.contains("fb.com") {
                    "facebook.com"
                } else if original.contains("tiktok.com") {
                    "tiktok.com"
                } else if original.contains("youtube.com") {
                    "youtube.com"
                } else if original.contains("reddit.com") {
                    "reddit.com"
                } else {
                    "example.com"
                };

                // Try to extract and link username from URL
                let username = if let Some(last_slash) = original.rfind('/') {
                    let user_part = &original[last_slash + 1..];
                    let user_part = user_part
                        .trim_start_matches('@')
                        .split('?')
                        .next()
                        .unwrap_or(user_part);

                    let (replaced, was_linked) = self.replace_with_components(user_part);
                    if was_linked {
                        replaced
                            .to_lowercase()
                            .replace(' ', "_")
                            .chars()
                            .filter(|c| c.is_alphanumeric() || *c == '_')
                            .collect::<String>()
                    } else {
                        let username: String = Username().fake_with_rng(&mut self.rng);
                        username
                    }
                } else {
                    let username: String = Username().fake_with_rng(&mut self.rng);
                    username
                };

                format!("https://{platform}/{username}")
            }
            "person" | "name" => {
                // Generate a realistic fake name
                // Note: component mappings are handled in fake_replacement_with_components
                let name: String = Name().fake_with_rng(&mut self.rng);
                name
            }
            "first_name" => {
                let name: String = FirstName().fake_with_rng(&mut self.rng);
                name
            }
            "last_name" => {
                let name: String = LastName().fake_with_rng(&mut self.rng);
                name
            }
            "company" | "organization" => {
                let company: String = CompanyName().fake_with_rng(&mut self.rng);
                company
            }
            "street_address" => {
                let street: String = StreetName().fake_with_rng(&mut self.rng);
                let num: u32 = self.rng.random_range(1..9999);
                format!("{num} {street}")
            }
            "city" => {
                let city: String = CityName().fake_with_rng(&mut self.rng);
                city
            }
            "state" => {
                let state: String = StateAbbr().fake_with_rng(&mut self.rng);
                state
            }
            "zip_code" | "postal_code" => {
                let zip: String = ZipCode().fake_with_rng(&mut self.rng);
                zip
            }
            "country" => {
                // Generate a fake country name
                // Using a list of fictional/common countries for consistency
                let countries = [
                    "Atlantis", "Narnia", "Wakanda", "Genovia", "Zamunda",
                    "Florin", "Guilder", "Latveria", "Sokovia", "Kahndaq",
                ];
                let idx = self.rng.random_range(0..countries.len());
                countries[idx].to_string()
            }
            "number" | "street_number" | "building_number" => {
                // Generate a fake street/building number
                let num: u32 = self.rng.random_range(1..9999);
                num.to_string()
            }
            "date" | "date_of_birth" => {
                // Generate a fake date (in past for DOB)
                let year: u32 = self.rng.random_range(1950..2005);
                let month: u32 = self.rng.random_range(1..13);
                let day: u32 = self.rng.random_range(1..29);
                format!("{year}-{month:02}-{day:02}")
            }
            "time" => {
                // Generate a fake time
                let hour: u32 = self.rng.random_range(0..24);
                let minute: u32 = self.rng.random_range(0..60);
                format!("{hour:02}:{minute:02}")
            }
            "username" => {
                // Generate a fake username
                let username: String = Username().fake_with_rng(&mut self.rng);
                let num: u32 = self.rng.random_range(1..999);
                format!("{username}{num}")
            }
            "location" => {
                // Generic location - use city
                let city: String = CityName().fake_with_rng(&mut self.rng);
                city
            }
            _ => {
                // Fallback for truly unknown patterns: generate fake text that looks plausible
                // Match the approximate length of the original
                let original_len = pii_match.matched_text.len();
                if original_len <= 10 {
                    // Short text - generate a word
                    let words = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"];
                    let idx = self.rng.random_range(0..words.len());
                    words[idx].to_string()
                } else if original_len <= 30 {
                    // Medium text - generate a phrase
                    let first: String = FirstName().fake_with_rng(&mut self.rng);
                    let last: String = LastName().fake_with_rng(&mut self.rng);
                    format!("{first} {last}")
                } else {
                    // Long text - generate an address-like string
                    let num: u32 = self.rng.random_range(1..9999);
                    let street: String = StreetName().fake_with_rng(&mut self.rng);
                    let city: String = CityName().fake_with_rng(&mut self.rng);
                    format!("{num} {street}, {city}")
                }
            }
        }
    }

    /// Normalize text for comparison (lowercase, umlaut expansion, etc.)
    ///
    /// Handles:
    /// - Case normalization (Müller → mueller)
    /// - German umlaut expansion (ü → ue, ö → oe, ä → ae, ß → ss)
    /// - Accent stripping (é → e, ñ → n)
    fn normalize_text(s: &str) -> String {
        let mut result = String::with_capacity(s.len() * 2);
        for c in s.to_lowercase().chars() {
            match c {
                'ä' => result.push_str("ae"),
                'ö' => result.push_str("oe"),
                'ü' => result.push_str("ue"),
                'ß' => result.push_str("ss"),
                'á' | 'à' | 'â' | 'ã' => result.push('a'),
                'é' | 'è' | 'ê' | 'ë' => result.push('e'),
                'í' | 'ì' | 'î' | 'ï' => result.push('i'),
                'ó' | 'ò' | 'ô' | 'õ' => result.push('o'),
                'ú' | 'ù' | 'û' => result.push('u'),
                'ñ' => result.push('n'),
                'ç' => result.push('c'),
                _ => result.push(c),
            }
        }
        result
    }

    /// Register a component mapping for derived PII linking.
    ///
    /// This enables linking between e.g., a person's name and their email.
    fn register_component(&mut self, original: &str, replacement: &str) {
        let normalized_orig = Self::normalize_text(original);
        let normalized_repl = Self::normalize_text(replacement);

        // Only register if both are meaningful (not empty, not too short)
        if normalized_orig.len() >= 2 && normalized_repl.len() >= 2 {
            self.component_cache
                .insert(normalized_orig, normalized_repl);
        }
    }

    /// Look up a component replacement by normalized original text.
    fn lookup_component(&self, original: &str) -> Option<&String> {
        let normalized = Self::normalize_text(original);
        self.component_cache.get(&normalized)
    }

    /// Try to replace parts of text using known component mappings.
    ///
    /// Handles text with separators like underscores, dots, or hyphens
    /// (e.g., "john_smith" or "john.smith" will match "john" and "smith").
    ///
    /// Returns (replaced_text, was_modified).
    fn replace_with_components(&self, text: &str) -> (String, bool) {
        // First, try direct substring replacement for compound matches
        let (result, modified) = self.replace_components_direct(text);

        if modified {
            return (result, true);
        }

        // If no direct match, try splitting on common separators and replacing parts
        self.replace_components_split(text)
    }

    /// Direct substring replacement in text.
    fn replace_components_direct(&self, text: &str) -> (String, bool) {
        let mut result = text.to_string();
        let mut modified = false;

        // Sort components by length (longest first) to avoid partial replacements
        let mut components: Vec<_> = self.component_cache.iter().collect();
        components.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        for (orig, repl) in components {
            // Case-insensitive replacement
            let normalized_result = Self::normalize_text(&result);
            if let Some(pos) = normalized_result.find(orig.as_str()) {
                // Find the actual position in the original string
                // We need to map from normalized position back to original
                let before_normalized = &normalized_result[..pos];

                // Count how many chars we need to skip in the original
                let mut orig_pos = 0;
                let mut norm_count = 0;
                for c in result.chars() {
                    if norm_count >= before_normalized.len() {
                        break;
                    }
                    let expansion_len = match c.to_lowercase().next().unwrap_or(c) {
                        'ä' | 'ö' | 'ü' => 2,
                        'ß' => 2,
                        _ => 1,
                    };
                    norm_count += expansion_len;
                    orig_pos += c.len_utf8();
                }

                // Find the end position similarly
                let mut end_pos = orig_pos;
                let mut matched_norm_len = 0;
                for c in result[orig_pos..].chars() {
                    if matched_norm_len >= orig.len() {
                        break;
                    }
                    let expansion_len = match c.to_lowercase().next().unwrap_or(c) {
                        'ä' | 'ö' | 'ü' => 2,
                        'ß' => 2,
                        _ => 1,
                    };
                    matched_norm_len += expansion_len;
                    end_pos += c.len_utf8();
                }

                // Preserve original case pattern if possible
                let original_part = &result[orig_pos..end_pos];
                let replacement = Self::match_case_pattern(original_part, repl);

                result = format!("{}{}{}", &result[..orig_pos], replacement, &result[end_pos..]);
                modified = true;
            }
        }

        (result, modified)
    }

    /// Split text on separators and replace each part individually.
    fn replace_components_split(&self, text: &str) -> (String, bool) {
        // Common separators in usernames/handles
        let separators = ['_', '.', '-'];

        // Check if text contains any separators
        let has_separator = text.chars().any(|c| separators.contains(&c));
        if !has_separator {
            return (text.to_string(), false);
        }

        let mut result = String::new();
        let mut current_part = String::new();
        let mut modified = false;

        for c in text.chars() {
            if separators.contains(&c) {
                // Process the current part
                if !current_part.is_empty() {
                    let normalized = Self::normalize_text(&current_part);
                    if let Some(replacement) = self.component_cache.get(&normalized) {
                        let replaced = Self::match_case_pattern(&current_part, replacement);
                        result.push_str(&replaced);
                        modified = true;
                    } else {
                        result.push_str(&current_part);
                    }
                    current_part.clear();
                }
                // Keep the separator
                result.push(c);
            } else {
                current_part.push(c);
            }
        }

        // Process the last part
        if !current_part.is_empty() {
            let normalized = Self::normalize_text(&current_part);
            if let Some(replacement) = self.component_cache.get(&normalized) {
                let replaced = Self::match_case_pattern(&current_part, replacement);
                result.push_str(&replaced);
                modified = true;
            } else {
                result.push_str(&current_part);
            }
        }

        (result, modified)
    }

    /// Match the case pattern of the original to the replacement.
    ///
    /// Examples:
    /// - "JOHN" + "donald" → "DONALD"
    /// - "John" + "donald" → "Donald"
    /// - "john" + "Donald" → "donald"
    fn match_case_pattern(original: &str, replacement: &str) -> String {
        if original.is_empty() || replacement.is_empty() {
            return replacement.to_string();
        }

        let orig_chars: Vec<char> = original.chars().collect();

        // Check if all uppercase
        if orig_chars.iter().all(|c| c.is_uppercase() || !c.is_alphabetic()) {
            return replacement.to_uppercase();
        }

        // Check if all lowercase
        if orig_chars.iter().all(|c| c.is_lowercase() || !c.is_alphabetic()) {
            return replacement.to_lowercase();
        }

        // Check if title case (first letter upper, rest lower)
        if orig_chars[0].is_uppercase()
            && orig_chars[1..].iter().all(|c| c.is_lowercase() || !c.is_alphabetic())
        {
            let mut result: String = replacement.to_lowercase();
            if let Some(first) = result.chars().next() {
                result = first.to_uppercase().to_string() + &result[first.len_utf8()..];
            }
            return result;
        }

        // Default: preserve replacement as-is
        replacement.to_string()
    }

    /// Check if a name part is an initial (e.g., "K.", "K", "J.R.")
    fn is_initial(s: &str) -> bool {
        let s = s.trim_end_matches('.');
        // Single letter or multiple initials like "J.R" or "JR"
        s.len() <= 3 && s.chars().all(|c| c.is_ascii_uppercase() || c == '.')
    }

    /// Build a name replacement preserving structure (initials, suffixes, etc.)
    ///
    /// Handles cases like:
    /// - "John Smith" → "Donald Duck"
    /// - "James K. Smith" → "Donald R. Duck" (preserves initial format)
    /// - "J. Robert Smith" → "D. Robert Duck"
    /// - "John Smith Jr." → "Donald Duck Jr." (preserves suffix)
    /// - "Dr. John Smith" → "Dr. Donald Duck" (preserves prefix)
    fn build_name_replacement(
        &mut self,
        original_parts: &[&str],
        fake_first: &str,
        fake_last: &str,
    ) -> (String, Vec<ComponentMapping>) {
        let mut components = Vec::new();
        let mut result_parts: Vec<String> = Vec::new();

        // Common prefixes and suffixes to preserve as-is
        let prefixes = [
            "Mr.", "Mrs.", "Ms.", "Miss", "Dr.", "Prof.", "Rev.", "Sr.", "Jr.",
        ];
        let suffixes = [
            "Jr.", "Jr", "Sr.", "Sr", "II", "III", "IV", "V", "PhD", "Ph.D.", "MD", "M.D.", "Esq.",
            "Esq",
        ];

        if original_parts.is_empty() {
            return (fake_first.to_string(), components);
        }

        if original_parts.len() == 1 {
            // Single name
            let part = original_parts[0];
            if Self::is_initial(part) {
                // Single initial - use first letter of fake_first
                let fake_initial = format!(
                    "{}{}",
                    fake_first.chars().next().unwrap_or('X'),
                    if part.ends_with('.') { "." } else { "" }
                );
                components.push(ComponentMapping {
                    original: part.to_string(),
                    replacement: fake_initial.clone(),
                    component_type: "initial".to_string(),
                });
                return (fake_initial, components);
            } else {
                components.push(ComponentMapping {
                    original: part.to_string(),
                    replacement: fake_first.to_string(),
                    component_type: "name".to_string(),
                });
                return (fake_first.to_string(), components);
            }
        }

        // Track which fake names we've used
        let mut used_first = false;
        let mut used_last = false;

        // Find indices of actual name parts (excluding prefixes/suffixes)
        let mut name_start = 0;
        let mut name_end = original_parts.len();

        // Check for prefix
        if prefixes
            .iter()
            .any(|p| original_parts[0].eq_ignore_ascii_case(p))
        {
            result_parts.push(original_parts[0].to_string());
            name_start = 1;
        }

        // Check for suffix
        if name_end > name_start
            && suffixes
                .iter()
                .any(|s| original_parts[name_end - 1].eq_ignore_ascii_case(s))
        {
            name_end -= 1;
        }

        // Process the actual name parts
        let name_parts = &original_parts[name_start..name_end];

        for (i, part) in name_parts.iter().enumerate() {
            let is_first_name_pos = i == 0;
            let is_last_name_pos = i == name_parts.len() - 1;

            if Self::is_initial(part) {
                // This is an initial - generate a matching fake initial
                let source_name = if !used_first {
                    fake_first
                } else if !used_last {
                    fake_last
                } else {
                    // Generate another name for additional initials
                    let extra: String = FirstName().fake_with_rng(&mut self.rng);
                    &extra.chars().next().unwrap_or('X').to_string()
                };

                let fake_initial = format!(
                    "{}{}",
                    source_name.chars().next().unwrap_or('X'),
                    if part.ends_with('.') { "." } else { "" }
                );

                components.push(ComponentMapping {
                    original: part.to_string(),
                    replacement: fake_initial.clone(),
                    component_type: if is_first_name_pos {
                        "first_initial".to_string()
                    } else {
                        "middle_initial".to_string()
                    },
                });

                // Also map the letter alone (without period) for flexibility
                let letter_only = part.trim_end_matches('.');
                if letter_only != *part {
                    let fake_letter = fake_initial.trim_end_matches('.').to_string();
                    components.push(ComponentMapping {
                        original: letter_only.to_string(),
                        replacement: fake_letter,
                        component_type: "initial_letter".to_string(),
                    });
                }

                result_parts.push(fake_initial);
            } else if is_first_name_pos && !used_first {
                // First full name
                components.push(ComponentMapping {
                    original: part.to_string(),
                    replacement: fake_first.to_string(),
                    component_type: "first_name".to_string(),
                });
                result_parts.push(fake_first.to_string());
                used_first = true;
            } else if is_last_name_pos && !used_last {
                // Last name
                components.push(ComponentMapping {
                    original: part.to_string(),
                    replacement: fake_last.to_string(),
                    component_type: "last_name".to_string(),
                });
                result_parts.push(fake_last.to_string());
                used_last = true;
            } else {
                // Middle name - generate a new fake name
                let fake_middle: String = FirstName().fake_with_rng(&mut self.rng);
                components.push(ComponentMapping {
                    original: part.to_string(),
                    replacement: fake_middle.clone(),
                    component_type: "middle_name".to_string(),
                });
                result_parts.push(fake_middle);
            }
        }

        // Add suffix if present
        if name_end < original_parts.len() {
            result_parts.push(original_parts[name_end].to_string());
        }

        (result_parts.join(" "), components)
    }

    /// Generate a fake replacement with component mappings for partial deanonymization.
    ///
    /// For names like "John Smith" → "Donald Duck", this also generates:
    /// - "John" → "Donald" (first_name)
    /// - "Smith" → "Duck" (last_name)
    ///
    /// This enables deanonymization when an LLM uses only part of the name.
    ///
    /// Uses a cache to ensure the same input always produces the same output.
    fn fake_replacement_with_components(
        &mut self,
        pii_match: &PiiMatch,
    ) -> (String, Vec<ComponentMapping>) {
        // Normalize text for cache lookup (lowercase, trimmed)
        let cache_key = pii_match.matched_text.to_lowercase();

        // Check cache first - same text should always get same replacement
        if let Some(cached) = self.replacement_cache.get(&cache_key) {
            return (cached.replacement.clone(), cached.components.clone());
        }

        // Generate new replacement
        let (replacement, components) = self.generate_fake_replacement(pii_match);

        // Cache it for consistency
        self.replacement_cache.insert(
            cache_key,
            CachedReplacement {
                replacement: replacement.clone(),
                components: components.clone(),
            },
        );

        (replacement, components)
    }

    /// Internal method to generate fake replacement (not cached).
    fn generate_fake_replacement(
        &mut self,
        pii_match: &PiiMatch,
    ) -> (String, Vec<ComponentMapping>) {
        let mut components = Vec::new();

        match pii_match.pattern_name.as_str() {
            "person" | "name" => {
                // Parse original name into components
                let original_parts: Vec<&str> = pii_match.matched_text.split_whitespace().collect();

                // Generate fake first and last names
                let fake_first: String = FirstName().fake_with_rng(&mut self.rng);
                let fake_last: String = LastName().fake_with_rng(&mut self.rng);

                // Build the replacement name preserving structure (initials, etc.)
                let (full_name, name_components) =
                    self.build_name_replacement(&original_parts, &fake_first, &fake_last);

                // Register all name components for derived PII linking
                for comp in &name_components {
                    self.register_component(&comp.original, &comp.replacement);
                }

                components.extend(name_components);
                (full_name, components)
            }
            "street_address" => {
                // Parse address components
                let original = &pii_match.matched_text;
                let fake_street: String = StreetName().fake_with_rng(&mut self.rng);
                let fake_num: u32 = self.rng.random_range(1..9999);
                let full_address = format!("{fake_num} {fake_street}");

                // Try to extract and map number and street separately
                if let Some(first_space) = original.find(' ') {
                    let (orig_num, orig_street) = original.split_at(first_space);
                    let orig_street = orig_street.trim();

                    // Only add component if the number looks like a number
                    if orig_num.chars().all(|c| c.is_ascii_digit()) {
                        components.push(ComponentMapping {
                            original: orig_num.to_string(),
                            replacement: fake_num.to_string(),
                            component_type: "street_number".to_string(),
                        });
                    }

                    if !orig_street.is_empty() {
                        components.push(ComponentMapping {
                            original: orig_street.to_string(),
                            replacement: fake_street,
                            component_type: "street_name".to_string(),
                        });
                    }
                }

                (full_address, components)
            }
            "email" => {
                // Parse email into local part and domain
                let original = &pii_match.matched_text;

                if let Some(at_pos) = original.find('@') {
                    let orig_local = &original[..at_pos];
                    let orig_domain = &original[at_pos + 1..];
                    let domain = &self.config.email_domain;

                    // Try to replace local part using known name components
                    let (replaced_local, was_linked) = self.replace_with_components(orig_local);

                    let fake_local = if was_linked {
                        // Local part was derived from known names - use linked replacement
                        // Normalize for email: lowercase, replace spaces with dots/underscores
                        // Don't add session ID - linking already makes it traceable
                        replaced_local
                            .to_lowercase()
                            .replace(' ', ".")
                            .chars()
                            .filter(|c| c.is_alphanumeric() || *c == '.' || *c == '_' || *c == '-')
                            .collect::<String>()
                    } else {
                        // No link found - generate random username
                        let username: String = Username().fake_with_rng(&mut self.rng);
                        if let Some(ref session) = self.session_id {
                            format!("{username}.{session}")
                        } else {
                            username
                        }
                    };

                    // Also try to link the domain if it matches a known name
                    // (e.g., meisner.de → mueller.de)
                    let fake_domain = if let Some(domain_name) = orig_domain.split('.').next() {
                        if let Some(replacement) = self.lookup_component(domain_name).cloned() {
                            // Domain contains a known name - replace it
                            let domain_suffix = &orig_domain[domain_name.len()..];
                            format!(
                                "{}{}",
                                replacement.to_lowercase(),
                                domain_suffix.to_lowercase()
                            )
                        } else {
                            // Use configured domain
                            domain.clone()
                        }
                    } else {
                        domain.clone()
                    };

                    let fake_email = format!("{fake_local}@{fake_domain}");

                    components.push(ComponentMapping {
                        original: orig_local.to_string(),
                        replacement: fake_local,
                        component_type: "email_local".to_string(),
                    });

                    (fake_email, components)
                } else {
                    // Malformed email - just generate a fake one
                    let username: String = Username().fake_with_rng(&mut self.rng);
                    let domain = &self.config.email_domain;
                    let fake_email = format!("{username}@{domain}");
                    (fake_email, components)
                }
            }
            "social_handle" | "twitter_handle" | "instagram_handle" => {
                // Parse social handle and try to link to known names
                let original = &pii_match.matched_text;
                let handle_part = original.trim_start_matches('@');

                // Try to replace using known name components
                let (replaced, was_linked) = self.replace_with_components(handle_part);

                let fake_handle = if was_linked {
                    // Use linked name parts - normalize for handle format
                    let normalized = replaced
                        .to_lowercase()
                        .replace(' ', "_")
                        .chars()
                        .filter(|c| c.is_alphanumeric() || *c == '_')
                        .collect::<String>();
                    format!("@{normalized}")
                } else {
                    // Generate random username
                    let username: String = Username().fake_with_rng(&mut self.rng);
                    let num: u32 = self.rng.random_range(0..999);
                    format!("@{username}{num}")
                };

                // Add component mapping for the handle
                components.push(ComponentMapping {
                    original: handle_part.to_string(),
                    replacement: fake_handle.trim_start_matches('@').to_string(),
                    component_type: "handle".to_string(),
                });

                (fake_handle, components)
            }
            "social_url" => {
                // Parse social URL and try to link username to known names
                let original = &pii_match.matched_text;

                // Extract platform from URL
                let platform = if original.contains("twitter.com") || original.contains("x.com") {
                    "twitter.com"
                } else if original.contains("instagram.com") {
                    "instagram.com"
                } else if original.contains("github.com") {
                    "github.com"
                } else if original.contains("linkedin.com") {
                    "linkedin.com"
                } else if original.contains("facebook.com") || original.contains("fb.com") {
                    "facebook.com"
                } else if original.contains("tiktok.com") {
                    "tiktok.com"
                } else if original.contains("youtube.com") {
                    "youtube.com"
                } else if original.contains("reddit.com") {
                    "reddit.com"
                } else {
                    "example.com"
                };

                // Extract and link username from URL
                let (fake_username, orig_username) =
                    if let Some(last_slash) = original.rfind('/') {
                        let user_part = &original[last_slash + 1..];
                        let user_part = user_part
                            .trim_start_matches('@')
                            .split('?')
                            .next()
                            .unwrap_or(user_part);

                        let (replaced, was_linked) = self.replace_with_components(user_part);
                        let fake = if was_linked {
                            replaced
                                .to_lowercase()
                                .replace(' ', "_")
                                .chars()
                                .filter(|c| c.is_alphanumeric() || *c == '_')
                                .collect::<String>()
                        } else {
                            let username: String = Username().fake_with_rng(&mut self.rng);
                            username
                        };
                        (fake, user_part.to_string())
                    } else {
                        let username: String = Username().fake_with_rng(&mut self.rng);
                        (username, String::new())
                    };

                let fake_url = format!("https://{platform}/{fake_username}");

                // Add component mapping for the username in the URL
                if !orig_username.is_empty() {
                    components.push(ComponentMapping {
                        original: orig_username,
                        replacement: fake_username,
                        component_type: "social_username".to_string(),
                    });
                }

                (fake_url, components)
            }
            _ => {
                // For other types, use the standard fake_replacement
                (self.fake_replacement(pii_match), components)
            }
        }
    }
}

impl Default for Replacer {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use crate::engine::patterns::Confidence;
    use crate::engine::patterns::PiiCategory;

    use super::*;

    fn make_match(pattern_name: &str, text: &str) -> PiiMatch {
        PiiMatch {
            pattern_name: pattern_name.to_string(),
            matched_text: text.to_string(),
            start: 0,
            end: text.len(),
            confidence: Confidence::High,
            category: PiiCategory::Contact,
        }
    }

    #[test]
    fn test_placeholder_replacement() {
        let mut replacer = Replacer::new(ReplacerConfig::placeholder());

        let email_match = make_match("email", "test@example.com");
        let result = replacer.replace(&email_match);
        assert_eq!(result.replacement, "<EMAIL>");

        let phone_match = make_match("phone_us", "555-123-4567");
        let result = replacer.replace(&phone_match);
        assert_eq!(result.replacement, "<PHONE>");
    }

    #[test]
    fn test_mask_replacement() {
        let mut replacer = Replacer::new(ReplacerConfig::mask());

        let email_match = make_match("email", "test@example.com");
        let result = replacer.replace(&email_match);
        assert_eq!(result.replacement, "XXXX@XXXXXXX.XXX");

        let ssn_match = make_match("ssn", "123-45-6789");
        let result = replacer.replace(&ssn_match);
        assert_eq!(result.replacement, "XXX-XX-XXXX");
    }

    #[test]
    fn test_consistent_replacement() {
        let mut replacer = Replacer::new(ReplacerConfig::consistent().with_seed(42));

        let match1 = make_match("email", "test@example.com");
        let match2 = make_match("email", "test@example.com");
        let match3 = make_match("email", "other@example.com");

        let result1 = replacer.replace(&match1);
        let result2 = replacer.replace(&match2);
        let result3 = replacer.replace(&match3);

        // Same input should get same output
        assert_eq!(result1.replacement, result2.replacement);

        // Different input should get different output
        assert_ne!(result1.replacement, result3.replacement);
    }

    #[test]
    fn test_replace_all() {
        let mut replacer = Replacer::new(ReplacerConfig::placeholder());

        let text = "Contact: test@example.com or 555-123-4567";
        let matches = vec![
            PiiMatch {
                pattern_name: "email".to_string(),
                matched_text: "test@example.com".to_string(),
                start: 9,
                end: 25,
                confidence: Confidence::High,
                category: PiiCategory::Contact,
            },
            PiiMatch {
                pattern_name: "phone_us".to_string(),
                matched_text: "555-123-4567".to_string(),
                start: 29,
                end: 41,
                confidence: Confidence::High,
                category: PiiCategory::Contact,
            },
        ];

        let (result, replacements) = replacer.replace_all(text, &matches);

        assert_eq!(result, "Contact: <EMAIL> or <PHONE>");
        assert_eq!(replacements.len(), 2);
    }

    #[test]
    fn test_seeded_random_is_deterministic() {
        let email_match = make_match("email", "test@example.com");

        // Create two replacers with same seed and random strategy
        let mut r1 = Replacer::new(ReplacerConfig {
            strategy: ReplacementStrategy::Random,
            seed: Some(42),
            ..Default::default()
        });
        let mut r2 = Replacer::new(ReplacerConfig {
            strategy: ReplacementStrategy::Random,
            seed: Some(42),
            ..Default::default()
        });

        let result1 = r1.replace(&email_match);
        let result2 = r2.replace(&email_match);

        // Same seed should produce identical results
        assert_eq!(result1.replacement, result2.replacement);
    }

    #[test]
    fn test_fake_name_generates_components() {
        let mut replacer = Replacer::new(ReplacerConfig {
            strategy: ReplacementStrategy::Fake,
            seed: Some(42),
            ..Default::default()
        });

        // Test simple two-part name
        let name_match = make_match("person", "John Smith");
        let result = replacer.replace(&name_match);

        assert!(
            !result.components.is_empty(),
            "Should have component mappings"
        );

        // Should have first_name and last_name components
        let has_first = result
            .components
            .iter()
            .any(|c| c.component_type == "first_name" && c.original == "John");
        let has_last = result
            .components
            .iter()
            .any(|c| c.component_type == "last_name" && c.original == "Smith");

        assert!(has_first, "Should have first_name component");
        assert!(has_last, "Should have last_name component");
    }

    #[test]
    fn test_fake_name_with_initial_generates_components() {
        let mut replacer = Replacer::new(ReplacerConfig {
            strategy: ReplacementStrategy::Fake,
            seed: Some(42),
            ..Default::default()
        });

        // Test name with middle initial
        let name_match = make_match("person", "James K. Smith");
        let result = replacer.replace(&name_match);

        // Should preserve the initial format
        assert!(
            result.replacement.contains('.'),
            "Should preserve initial period"
        );

        // Should have component for the initial
        let has_initial = result
            .components
            .iter()
            .any(|c| c.component_type == "middle_initial" && c.original == "K.");
        assert!(has_initial, "Should have middle_initial component");

        // Should have first and last name
        let has_first = result
            .components
            .iter()
            .any(|c| c.component_type == "first_name");
        let has_last = result
            .components
            .iter()
            .any(|c| c.component_type == "last_name");

        assert!(has_first, "Should have first_name component");
        assert!(has_last, "Should have last_name component");
    }

    #[test]
    fn test_fake_name_with_suffix_preserves_suffix() {
        let mut replacer = Replacer::new(ReplacerConfig {
            strategy: ReplacementStrategy::Fake,
            seed: Some(42),
            ..Default::default()
        });

        // Test name with suffix
        let name_match = make_match("person", "John Smith Jr.");
        let result = replacer.replace(&name_match);

        // Should preserve the suffix
        assert!(
            result.replacement.ends_with("Jr."),
            "Should preserve Jr. suffix, got: {}",
            result.replacement
        );
    }

    #[test]
    fn test_fake_name_with_prefix_preserves_prefix() {
        let mut replacer = Replacer::new(ReplacerConfig {
            strategy: ReplacementStrategy::Fake,
            seed: Some(42),
            ..Default::default()
        });

        // Test name with prefix
        let name_match = make_match("person", "Dr. John Smith");
        let result = replacer.replace(&name_match);

        // Should preserve the prefix
        assert!(
            result.replacement.starts_with("Dr."),
            "Should preserve Dr. prefix, got: {}",
            result.replacement
        );
    }

    #[test]
    fn test_email_links_to_name() {
        let mut replacer = Replacer::new(ReplacerConfig {
            strategy: ReplacementStrategy::Fake,
            seed: Some(42),
            email_domain: "example.com".to_string(),
        });

        // First process a name
        let name_match = make_match("person", "Hansgerd Meisner");
        let name_result = replacer.replace(&name_match);

        // Extract the fake first and last names
        let fake_parts: Vec<&str> = name_result.replacement.split_whitespace().collect();
        assert!(fake_parts.len() >= 2, "Should have at least first and last name");
        let fake_first = fake_parts[0].to_lowercase();
        let fake_last = fake_parts[fake_parts.len() - 1].to_lowercase();

        // Now process an email that contains the original name parts
        let email_match = make_match("email", "hansgerd@meisner.de");
        let email_result = replacer.replace(&email_match);

        // Email should contain the linked fake names
        let email_lower = email_result.replacement.to_lowercase();
        assert!(
            email_lower.contains(&fake_first),
            "Email '{}' should contain fake first name '{}' (from name '{}')",
            email_result.replacement,
            fake_first,
            name_result.replacement
        );
        assert!(
            email_lower.contains(&fake_last),
            "Email '{}' should contain fake last name '{}' (from name '{}')",
            email_result.replacement,
            fake_last,
            name_result.replacement
        );
    }

    #[test]
    fn test_email_links_with_umlaut_normalization() {
        let mut replacer = Replacer::new(ReplacerConfig {
            strategy: ReplacementStrategy::Fake,
            seed: Some(42),
            email_domain: "example.com".to_string(),
        });

        // Process a name with umlauts
        let name_match = make_match("person", "Hans Müller");
        let name_result = replacer.replace(&name_match);

        // Extract the fake last name
        let fake_parts: Vec<&str> = name_result.replacement.split_whitespace().collect();
        let fake_last = fake_parts.last().unwrap().to_lowercase();

        // Process an email with ASCII-ized version of the name (Mueller instead of Müller)
        let email_match = make_match("email", "hans@mueller.de");
        let email_result = replacer.replace(&email_match);

        // Email should contain the linked fake last name
        assert!(
            email_result.replacement.to_lowercase().contains(&fake_last),
            "Email '{}' should contain fake last name '{}' (Mueller normalized to match Müller)",
            email_result.replacement,
            fake_last
        );
    }

    #[test]
    fn test_normalize_text() {
        assert_eq!(Replacer::normalize_text("Müller"), "mueller");
        assert_eq!(Replacer::normalize_text("MÜLLER"), "mueller");
        assert_eq!(Replacer::normalize_text("Schröder"), "schroeder");
        assert_eq!(Replacer::normalize_text("Größe"), "groesse");
        assert_eq!(Replacer::normalize_text("Bär"), "baer");
        assert_eq!(Replacer::normalize_text("José"), "jose");
        assert_eq!(Replacer::normalize_text("François"), "francois");
    }

    #[test]
    fn test_match_case_pattern() {
        assert_eq!(Replacer::match_case_pattern("JOHN", "donald"), "DONALD");
        assert_eq!(Replacer::match_case_pattern("john", "Donald"), "donald");
        assert_eq!(Replacer::match_case_pattern("John", "donald"), "Donald");
        assert_eq!(Replacer::match_case_pattern("Smith", "duck"), "Duck");
    }
}
