//! JSON field/path selection.
//!
//! nym's generic JSON anonymization walks every string value, including
//! structural identifiers such as `id`/`parentId` UUIDs. For session/teacher
//! datasets the role/op/fact/provenance relationships must survive while the
//! text and identifying metadata are audited. This module lets callers select
//! which JSON paths are scanned (include) and which are skipped (exclude),
//! and report which paths were covered versus skipped.

use std::collections::BTreeSet;

/// A parsed path segment.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Segment {
    /// A literal object key.
    Key(String),
    /// A numeric array index.
    Index(usize),
    /// A wildcard matching one segment (object key or any array index).
    Wildcard,
    /// A recursive wildcard matching zero or more segments.
    Recursive,
}

/// A compiled path selector pattern (e.g. `users[*].email`).
#[derive(Debug, Clone)]
pub struct PathPattern {
    segments: Vec<Segment>,
    /// Whether the pattern is the trivial root selector `*`.
    catch_all: bool,
}

impl PathPattern {
    /// Compile a selector string. Accepts dot-separated keys, array indices
    /// (`users[0]`, `[0]`, `users[]`, `users[*]`), `*` for one segment,
    /// and `**` for zero or more segments (e.g. `**.id`).
    pub fn parse(s: &str) -> Result<Self, String> {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return Err("empty path selector".to_string());
        }
        if trimmed == "*" {
            return Ok(Self {
                segments: Vec::new(),
                catch_all: true,
            });
        }

        let mut segments = Vec::new();
        let mut key_buf = String::new();
        let mut chars = trimmed.chars().peekable();

        while let Some(c) = chars.next() {
            match c {
                '.' => {
                    if !key_buf.is_empty() {
                        segments.push(Segment::Key(key_buf.clone()));
                        key_buf.clear();
                    }
                }
                '*' => {
                    // A standalone `*` is a wildcard segment.
                    if !key_buf.is_empty() {
                        segments.push(Segment::Key(key_buf.clone()));
                        key_buf.clear();
                    }
                    if chars.peek() == Some(&'*') {
                        chars.next();
                        segments.push(Segment::Recursive);
                    } else {
                        segments.push(Segment::Wildcard);
                    }
                }
                '[' => {
                    // Flush any pending key.
                    if !key_buf.is_empty() {
                        segments.push(Segment::Key(key_buf.clone()));
                        key_buf.clear();
                    }
                    // Read until ']'.
                    let mut inner = String::new();
                    let mut closed = false;
                    for c2 in chars.by_ref() {
                        if c2 == ']' {
                            closed = true;
                            break;
                        }
                        inner.push(c2);
                    }
                    if !closed {
                        return Err(format!("unterminated '[' in selector '{s}'"));
                    }
                    let inner = inner.trim();
                    if inner.is_empty() || inner == "*" {
                        segments.push(Segment::Wildcard);
                    } else {
                        let idx: usize = inner
                            .parse()
                            .map_err(|_| format!("invalid array index '{inner}' in '{s}'"))?;
                        segments.push(Segment::Index(idx));
                    }
                }
                _ => key_buf.push(c),
            }
        }
        if !key_buf.is_empty() {
            segments.push(Segment::Key(key_buf));
        }
        if segments.is_empty() {
            return Err(format!("path selector '{s}' has no segments"));
        }
        Ok(Self {
            segments,
            catch_all: false,
        })
    }

    /// Whether this pattern matches the given path string.
    pub fn matches(&self, path: &str) -> bool {
        self.match_path(path, false)
    }

    /// Whether this pattern matches the given path or a strict prefix of it
    /// (i.e. the path is a descendant of the selected path).
    pub fn matches_subtree(&self, path: &str) -> bool {
        self.match_path(path, true)
    }

    fn match_path(&self, path: &str, subtree: bool) -> bool {
        if self.catch_all {
            return true;
        }
        let path_segments = parse_path_segments(path);
        segments_match(&self.segments, &path_segments, subtree)
    }
}

/// Parse an nym-generated value path (e.g. `users[0].email`, `(root)`) into
/// segments. Unknown bracket forms are tolerated as wildcards so a selector
/// never silently fails to match on a legitimately selected path.
fn parse_path_segments(path: &str) -> Vec<Segment> {
    if path == "(root)" || path.is_empty() {
        return Vec::new();
    }
    let mut segments = Vec::new();
    let mut key_buf = String::new();
    let mut chars = path.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '.' => {
                if !key_buf.is_empty() {
                    segments.push(Segment::Key(std::mem::take(&mut key_buf)));
                }
            }
            '[' => {
                if !key_buf.is_empty() {
                    segments.push(Segment::Key(std::mem::take(&mut key_buf)));
                }
                let mut inner = String::new();
                let mut closed = false;
                for c2 in chars.by_ref() {
                    if c2 == ']' {
                        closed = true;
                        break;
                    }
                    inner.push(c2);
                }
                if !closed {
                    // Malformed; treat the remainder as a literal key segment.
                    segments.push(Segment::Key(inner));
                    break;
                }
                let inner = inner.trim();
                if let Ok(idx) = inner.parse::<usize>() {
                    segments.push(Segment::Index(idx));
                } else {
                    segments.push(Segment::Wildcard);
                }
            }
            _ => key_buf.push(c),
        }
    }
    if !key_buf.is_empty() {
        segments.push(Segment::Key(key_buf));
    }
    segments
}

/// Track reachable path prefixes in O(pattern × path) time and O(path) space.
/// This avoids exponential backtracking with consecutive recursive wildcards.
fn segments_match(pattern: &[Segment], path: &[Segment], subtree: bool) -> bool {
    let mut reachable = vec![false; path.len() + 1];
    reachable[0] = true;
    for segment in pattern {
        let mut next = vec![false; path.len() + 1];
        if *segment == Segment::Recursive {
            next[0] = reachable[0];
            for index in 1..=path.len() {
                next[index] = reachable[index] || next[index - 1];
            }
        } else {
            for (index, actual) in path.iter().enumerate() {
                next[index + 1] = reachable[index] && segment_matches(segment, actual);
            }
        }
        reachable = next;
    }
    if subtree {
        reachable.into_iter().any(|matched| matched)
    } else {
        reachable[path.len()]
    }
}

/// Match a single concrete path step; recursion is handled by the prefix walk.
fn segment_matches(pattern: &Segment, actual: &Segment) -> bool {
    match (pattern, actual) {
        (Segment::Wildcard, _) | (Segment::Index(_), Segment::Wildcard) => true,
        (Segment::Key(key), Segment::Key(other)) => key == other,
        (Segment::Index(index), Segment::Index(other)) => index == other,
        _ => false,
    }
}

/// A set of include/exclude path selectors controlling how JSON is scanned.
#[derive(Debug, Clone, Default)]
pub struct PathSelector {
    include: Vec<PathPattern>,
    exclude: Vec<PathPattern>,
}

impl PathSelector {
    /// Build a selector from include/exclude strings; any invalid pattern
    /// returns an error so misconfiguration fails loudly rather than being
    /// silently ignored.
    pub fn new(include: &[String], exclude: &[String]) -> Result<Self, String> {
        let include = include
            .iter()
            .map(|s| PathPattern::parse(s))
            .collect::<Result<Vec<_>, _>>()?;
        let exclude = exclude
            .iter()
            .map(|s| PathPattern::parse(s))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { include, exclude })
    }

    /// Whether this selector restricts scanning at all (i.e. it is not the
    /// pass-everything default).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn is_restrictive(&self) -> bool {
        !self.include.is_empty() || !self.exclude.is_empty()
    }

    /// Decide whether a value at `path` should be scanned.
    ///
    /// Semantics: if no include patterns are set, every path is eligible unless
    /// it matches an exclude. If include patterns are set, a path is eligible
    /// only when it matches an include and does not match an exclude.
    pub fn should_scan(&self, path: &str) -> bool {
        let included = if self.include.is_empty() {
            true
        } else {
            self.include.iter().any(|p| p.matches(path))
        };
        if !included {
            return false;
        }
        // Excludes cover the matched path and its descendants.
        !self.exclude.iter().any(|p| p.matches_subtree(path))
    }
}

/// Coverage report for a scan.
#[derive(Debug, Clone, Default)]
pub struct CoverageReport {
    /// Paths that were scanned (eligible for detection).
    pub scanned: BTreeSet<String>,
    /// Paths that were skipped by the selector.
    pub skipped: BTreeSet<String>,
}

impl CoverageReport {
    /// Number of paths scanned.
    pub fn scanned_count(&self) -> usize {
        self.scanned.len()
    }

    /// Number of paths skipped.
    pub fn skipped_count(&self) -> usize {
        self.skipped.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recursive_wildcard_zero_or_more_segments() {
        for (selector, yes, no) in [
            (
                "**.id",
                vec!["id", "meta.id", "messages[0].id", "資料[3].情報.id"],
                vec!["ids", "meta.id.child", "meta.identity"],
            ),
            (
                "messages.**.id",
                vec!["messages.id", "messages[0].id", "messages[0].meta.id"],
                vec!["meta.id", "messages[0].ids"],
            ),
            (
                "meta.**",
                vec!["meta", "meta.id", "meta[0].nested.id"],
                vec!["metadata.id", "user.meta"],
            ),
            (
                "**.**.id",
                vec!["id", "meta.id", "messages[0].meta.id"],
                vec!["id.child"],
            ),
            ("**", vec!["(root)", "id", "資料[3].情報.id"], vec![]),
        ] {
            let pattern = PathPattern::parse(selector).unwrap();
            for path in yes {
                assert!(pattern.matches(path), "{selector} should match {path}");
            }
            for path in no {
                assert!(!pattern.matches(path), "{selector} should not match {path}");
            }
        }
        assert!(
            !PathPattern::parse("*.id")
                .unwrap()
                .matches("messages[0].id")
        );
        let selector = PathSelector::new(&[], &["**.id".to_string()]).unwrap();
        assert!(!selector.should_scan("messages[0].id.child"));
        assert!(!selector.should_scan("meta.id"));
        assert!(selector.should_scan("meta.content"));
    }

    #[test]
    fn test_parse_dot_path() {
        let p = PathPattern::parse("user.email").unwrap();
        assert!(p.matches("user.email"));
        assert!(!p.matches("user.name"));
    }

    #[test]
    fn test_parse_array_index() {
        let p = PathPattern::parse("users[0].email").unwrap();
        assert!(p.matches("users[0].email"));
        assert!(!p.matches("users[1].email"));
    }

    #[test]
    fn test_parse_wildcard_index() {
        let p = PathPattern::parse("users[*].email").unwrap();
        assert!(p.matches("users[0].email"));
        assert!(p.matches("users[7].email"));
        assert!(!p.matches("users[0].name"));
    }

    #[test]
    fn test_wildcard_key() {
        let p = PathPattern::parse("*.email").unwrap();
        assert!(p.matches("user.email"));
        assert!(p.matches("admin.email"));
        assert!(!p.matches("user.name"));
    }

    #[test]
    fn test_catch_all() {
        let p = PathPattern::parse("*").unwrap();
        assert!(p.matches("anything.here"));
        assert!(p.matches("(root)"));
    }

    #[test]
    fn test_root_path() {
        // A root-level field path.
        let p = PathPattern::parse("email").unwrap();
        assert!(p.matches("email"));
        assert!(!p.matches("user.email"));
    }

    #[test]
    fn test_selector_include() {
        let sel =
            PathSelector::new(&["user.email".to_string(), "user.phone".to_string()], &[]).unwrap();
        assert!(sel.should_scan("user.email"));
        assert!(sel.should_scan("user.phone"));
        assert!(!sel.should_scan("user.name"));
        assert!(!sel.should_scan("user.email.extra"));
    }

    #[test]
    fn test_selector_exclude() {
        let sel = PathSelector::new(&[], &["log".to_string()]).unwrap();
        assert!(sel.should_scan("user.email"));
        assert!(!sel.should_scan("log"));
        assert!(!sel.should_scan("log[0]"));
    }

    #[test]
    fn test_selector_no_restriction() {
        let sel = PathSelector::new(&[], &[]).unwrap();
        assert!(!sel.is_restrictive());
        assert!(sel.should_scan("anything"));
    }

    #[test]
    fn test_invalid_selector_rejected() {
        assert!(PathPattern::parse("").is_err());
        assert!(PathPattern::parse("a[").is_err());
    }
}
