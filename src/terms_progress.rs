//! Stderr-only count progress; no corpus values, identifiers or paths.
use crate::terms::Builder;
use anyhow::Result;
use std::{
    io::{self, IsTerminal, Write},
    time::{Duration, Instant},
};

pub struct Scan<'a> {
    builder: &'a mut Builder,
    stage: &'static str,
    total: usize,
    files: usize,
    units: u64,
    enabled: bool,
    terminal: bool,
    started: Instant,
    drawn: Instant,
    width: usize,
    finished: bool,
}

impl<'a> Scan<'a> {
    pub fn new(builder: &'a mut Builder, stage: &'static str, total: usize, enabled: bool) -> Self {
        let now = Instant::now();
        let mut scan = Self {
            builder,
            stage,
            total: total.max(1),
            files: 0,
            units: 0,
            enabled,
            terminal: io::stderr().is_terminal()
                && std::env::var("TERM").is_ok_and(|v| v != "dumb"),
            started: now,
            drawn: now,
            width: 0,
            finished: false,
        };
        scan.draw("scanning");
        scan
    }

    pub fn end_record(&mut self) {
        self.builder.end_record();
    }

    pub fn parse_record(
        &mut self,
        source: &str,
        location: &str,
        raw: &str,
        number: usize,
    ) -> Result<serde_json::Value> {
        let parsed = crate::terms_unicode::parse(raw, number, self.builder.repairs_unicode())?;
        self.builder
            .bind_record(source, location, raw, parsed.replacements)?;
        Ok(parsed.value)
    }

    pub fn ingest(&mut self, source: &str, path: &str, text: &str) -> Result<()> {
        self.builder.ingest(source, path, text)?;
        self.units += 1;
        self.refresh();
        Ok(())
    }

    pub fn file_scanned(&mut self) {
        self.files += 1;
        self.refresh();
    }

    fn refresh(&mut self) {
        if self.enabled && self.terminal && self.drawn.elapsed() >= Duration::from_millis(125) {
            self.draw("scanning");
        }
    }

    pub fn finish(mut self) {
        self.finished = true;
        self.draw("scanned");
    }

    fn line(&self, status: &str, seconds: u64) -> String {
        let filled = self.files * 24 / self.total;
        format!(
            "[{}{}] {}: {}/{} files, {} text units, {}:{:02} elapsed ({status})",
            "#".repeat(filled),
            "-".repeat(24 - filled),
            self.stage,
            self.files,
            self.total,
            self.units,
            seconds / 60,
            seconds % 60
        )
    }

    fn draw(&mut self, status: &str) {
        if !self.enabled {
            return;
        }
        let line = self.line(status, self.started.elapsed().as_secs());
        let mut stderr = io::stderr().lock();
        if self.terminal {
            self.width = self.width.max(line.len());
            let _ = write!(stderr, "\r{line:width$}", width = self.width);
            if status != "scanning" {
                let _ = writeln!(stderr);
            }
            let _ = stderr.flush();
        } else {
            let _ = writeln!(stderr, "{line}");
        }
        self.drawn = Instant::now();
    }
}

impl Drop for Scan<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.draw("incomplete");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terms::Settings;
    #[test]
    fn rendering_uses_counts_only_and_units_advance_inside_a_file() {
        let mut builder = Builder::new(Settings::default(), None).unwrap();
        let mut scan = Scan::new(&mut builder, "1/2 Proposal", 4, false);
        scan.ingest("private-source", "private-path", "PrivateExample")
            .unwrap();
        assert_eq!(
            scan.line("scanning", 0),
            "[------------------------] 1/2 Proposal: 0/4 files, 1 text units, 0:00 elapsed (scanning)"
        );
        scan.file_scanned();
        assert_eq!(
            scan.line("scanning", 0),
            "[######------------------] 1/2 Proposal: 1/4 files, 1 text units, 0:00 elapsed (scanning)"
        );
        scan.finish();
    }
}
