//! Secret key for tokens derived from values.
//!
//! `--strategy hash` and `--placeholder-scope corpus` turn a value into a stable
//! token. Derived from a bare hash, a token is reversible by anyone who can
//! guess the value: hash a list of candidate names and compare. Tokens are
//! therefore HMAC-SHA256 under a secret key, so stability holds for whoever has
//! the key and guessing needs it too.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use sha2::{Digest, Sha256};

/// Bytes in a token key.
pub const KEY_BYTES: usize = 32;

static KEY: OnceLock<[u8; KEY_BYTES]> = OnceLock::new();

/// Install the process-wide token key. A second install with a different key
/// is an error: tokens from one run must not mix two keys.
pub fn install(key: [u8; KEY_BYTES]) -> Result<(), String> {
    let installed = KEY.get_or_init(|| key);
    if *installed == key {
        Ok(())
    } else {
        Err("a different token key is already installed".into())
    }
}

/// The installed key. Code that derives tokens from values runs only after the
/// command installed one; reaching this without a key is a bug, and failing is
/// the safe outcome (no unkeyed token is ever written).
#[cfg(not(test))]
fn key() -> &'static [u8; KEY_BYTES] {
    KEY.get()
        .expect("token key must be installed before deriving tokens from values")
}

#[cfg(test)]
fn key() -> &'static [u8; KEY_BYTES] {
    KEY.get_or_init(|| [7; KEY_BYTES])
}

/// HMAC-SHA256 of `message` under the installed key.
pub fn mac(message: &[u8]) -> [u8; 32] {
    hmac_sha256(key(), message)
}

/// HMAC-SHA256 (RFC 2104) on `sha2`, to avoid another dependency.
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    const BLOCK: usize = 64;
    let mut block = [0u8; BLOCK];
    if key.len() > BLOCK {
        block[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        block[..key.len()].copy_from_slice(key);
    }
    let pad = |byte: u8| block.map(|b| b ^ byte);
    let inner = Sha256::new()
        .chain_update(pad(0x36))
        .chain_update(message)
        .finalize();
    Sha256::new()
        .chain_update(pad(0x5c))
        .chain_update(inner)
        .finalize()
        .into()
}

/// Lowercase hex of `bytes`.
pub fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut out, b| {
            let _ = write!(out, "{b:02x}");
            out
        })
}

/// Short non-secret fingerprint of a key, for logs and summaries.
pub fn fingerprint(key: &[u8; KEY_BYTES]) -> String {
    hex(&hmac_sha256(key, b"nym token key id")[..4])
}

/// Default location: `$XDG_DATA_HOME/nym/token.key`.
pub fn default_path() -> PathBuf {
    crate::config::data_dir().join("token.key")
}

/// Load a key from an explicit file (which must exist), or from the default
/// location, creating it with a fresh random key on first use.
pub fn load(explicit: Option<&Path>) -> Result<[u8; KEY_BYTES], String> {
    if let Some(path) = explicit {
        return read(path);
    }
    let path = default_path();
    if !path.exists() {
        create(&path)?;
    }
    read(&path)
}

fn read(path: &Path) -> Result<[u8; KEY_BYTES], String> {
    let text = std::fs::read_to_string(path)
        .map_err(|_| format!("token key file {} cannot be read", path.display()))?;
    let hex = text.trim();
    if hex.len() != KEY_BYTES * 2 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!(
            "token key file {} must hold {} hex characters",
            path.display(),
            KEY_BYTES * 2
        ));
    }
    let mut key = [0u8; KEY_BYTES];
    for (i, byte) in key.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16)
            .map_err(|_| "token key is not hex".to_string())?;
    }
    Ok(key)
}

fn create(path: &Path) -> Result<(), String> {
    use fake::rand::{RngCore, SeedableRng, rngs::StdRng};
    use std::io::Write;
    let parent = path.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent).map_err(|_| format!("cannot create {}", parent.display()))?;
    let mut key = [0u8; KEY_BYTES];
    StdRng::from_os_rng().fill_bytes(&mut key);
    let text = hex(&key);
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    match options.open(path) {
        Ok(mut file) => file
            .write_all(format!("{text}\n").as_bytes())
            .and_then(|()| file.sync_all())
            .map_err(|_| format!("cannot write token key file {}", path.display())),
        // Another process created it first: use theirs.
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(_) => Err(format!("cannot create token key file {}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::{hex, hmac_sha256};

    #[test]
    fn hmac_matches_rfc_4231_vectors() {
        assert_eq!(
            hex(&hmac_sha256(&[0x0b; 20], b"Hi There")),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
        assert_eq!(
            hex(&hmac_sha256(b"Jefe", b"what do ya want for nothing?")),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        // Key longer than the block size is hashed first (test case 6).
        assert_eq!(
            hex(&hmac_sha256(
                &[0xaa; 131],
                b"Test Using Larger Than Block-Size Key - Hash Key First"
            )),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }
}
