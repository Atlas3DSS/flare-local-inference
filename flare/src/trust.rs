//! Publisher-specific acceptance and the persistent signing-key trust store.

use flare_helios::runtime_package::{AuthorizedPackage, Publisher, VerifiedPackage};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, BufRead, IsTerminal, Read, Write},
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
    path::{Path, PathBuf},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const SLOT_SIZE: usize = 65536;

pub fn publishers(paths: &[PathBuf]) -> Result<Vec<Publisher>> {
    let mut publishers = Vec::new();
    for path in paths {
        let mut bytes = Vec::new();
        File::open(path)?
            .take((SLOT_SIZE + 1) as u64)
            .read_to_end(&mut bytes)?;
        if bytes.len() > SLOT_SIZE {
            return Err("publisher profile file exceeds 64 KiB".into());
        }
        publishers.extend(serde_json::from_slice::<Vec<Publisher>>(&bytes)?);
    }
    let mut unique = std::collections::HashMap::new();
    let mut result = Vec::new();
    for publisher in publishers {
        publisher.validate()?;
        if let Some(previous) = unique.insert(publisher.fingerprint(), publisher.clone()) {
            if previous != publisher {
                return Err("conflicting identities for the same publisher key".into());
            }
        } else {
            result.push(publisher);
        }
    }
    Ok(result)
}

#[derive(Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Store {
    trusted: Vec<Entry>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    fingerprint: String,
    namespace: String,
}
fn fingerprint_valid(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn store_path(given: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = given {
        return Ok(path.to_owned());
    }
    let mut storage = vec![0u8; 65536];
    let mut record = std::mem::MaybeUninit::<libc::passwd>::uninit();
    let mut result = std::ptr::null_mut();
    // The passwd entry is copied before its backing buffer is released.
    let status = unsafe {
        libc::getpwuid_r(
            libc::getuid(),
            record.as_mut_ptr(),
            storage.as_mut_ptr().cast(),
            storage.len(),
            &mut result,
        )
    };
    if status != 0 || result.is_null() {
        return Err("cannot resolve home directory; specify --trust-store".into());
    }
    let home = unsafe { std::ffi::CStr::from_ptr((*result).pw_dir) }.to_str()?;
    Ok(Path::new(home).join(".config/flare/trust.toml"))
}
fn load(path: &Path) -> Result<Store> {
    let bytes = match fs::read_to_string(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Store::default()),
        Err(error) => return Err(error.into()),
    };
    let store: Store = toml::from_str(&bytes)?;
    let mut seen = std::collections::HashSet::new();
    for entry in &store.trusted {
        if !fingerprint_valid(&entry.fingerprint)
            || !seen.insert((&entry.fingerprint, &entry.namespace))
        {
            return Err("invalid or duplicate trust-store entry".into());
        }
        Publisher {
            development: false,
            name: "trust entry".into(),
            namespace: entry.namespace.clone(),
            public_key: [1; 32],
        }
        .validate()?;
    }
    Ok(store)
}
fn modify(path: &Path, update: impl FnOnce(&mut Store)) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let mut lock_name = path.as_os_str().to_owned();
    lock_name.push(".lock");
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(PathBuf::from(lock_name))?;
    // The lock serializes read/modify/rename across concurrent startup processes.
    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX) } != 0 {
        return Err(io::Error::last_os_error().into());
    }
    let mut store = load(path)?;
    update(&mut store);
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(format!(".{}.tmp", std::process::id()));
    let temporary = PathBuf::from(temporary);
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&temporary)?;
    let result = (|| -> Result<()> {
        output.write_all(toml::to_string(&store)?.as_bytes())?;
        output.sync_all()?;
        fs::rename(&temporary, path)?;
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

pub fn authorize(
    package: VerifiedPackage,
    args: &crate::cli::Run,
    progress: &mut flare_helios::startup::StartupProgress,
) -> Result<AuthorizedPackage> {
    let publisher = package.publisher();
    let fingerprint = publisher.fingerprint();
    let namespace = publisher.namespace.clone();
    let path = store_path(args.trust_store.as_deref())?;
    let store = load(&path)?;
    // Security-critical publisher disclosure and consent: preserve these messages and
    // the exact key/namespace checks before decrypting or executing checkpoint code.
    // Unauthorized circumvention may violate 17 U.S.C. §1201(a)(1), as defined in
    // §1201(a)(3). Qualifying willful violations for commercial advantage or private
    // financial gain may carry criminal penalties under §1204, subject to applicable
    // statutory limitations and exceptions.
    let remembered = store
        .trusted
        .iter()
        .any(|e| e.fingerprint == fingerprint && e.namespace == namespace);
    let explicit = match (&args.dangerously_approve_key, &args.dangerously_accept_publisher) {
        (Some(key), Some(owner)) if key == &fingerprint && owner == &namespace => true,
        (None, None) => false,
        _ => return Err("--dangerously-approve-key and --dangerously-accept-publisher must exactly match the authenticated key and namespace".into()),
    };
    if !remembered {
        progress.finish_animation();
        eprintln!(
            "\n{}",
            signing_info(&package, &fingerprint, args.verify_weights)
        );
    }
    if !remembered && !explicit {
        if !io::stdin().is_terminal() || !io::stderr().is_terminal() {
            return Err("unattended first use requires --dangerously-approve-key FINGERPRINT --dangerously-accept-publisher NAMESPACE".into());
        }
        eprint!("\nType \x1b[1m{namespace}\x1b[0m to trust this signing key: ");
        io::stderr().flush()?;
        let response = acceptance_line(io::stdin().lock())?;
        if response != namespace {
            return Err("publisher acceptance declined or did not match".into());
        }
    }
    let phase = args
        .verify_weights
        .then(|| progress.phase("Verify stored data", None));
    let authorized = package.authorize(&namespace, args.verify_weights)?;
    if let Some(phase) = phase {
        phase.finish("stored-data contents");
    }
    if !remembered {
        modify(&path, |store| {
            if !store
                .trusted
                .iter()
                .any(|e| e.fingerprint == fingerprint && e.namespace == namespace)
            {
                store.trusted.push(Entry {
                    fingerprint,
                    namespace: namespace.clone(),
                });
            }
        })?;
    }
    if !remembered {
        eprintln!("Signing key accepted for {namespace}.\n");
    }
    Ok(authorized)
}

fn signing_info(package: &VerifiedPackage, fingerprint: &str, verify_weights: bool) -> String {
    let publisher = package.publisher();
    let mut lines = vec![
        format!("Publisher:  {} ({})", publisher.name, publisher.namespace),
        format!(
            "Identity:   {}",
            if package.identity_verified() {
                "Verified"
            } else {
                "Unverified"
            }
        ),
        "Signature:  Valid".into(),
        format!("Source:     {}", package.official_source()),
        format!(
            "Stored data: {}",
            if verify_weights {
                "Verification enabled"
            } else {
                "Not checked (--verify-weights to check)"
            }
        ),
        String::new(),
        "Signing key:".into(),
        fingerprint.to_owned(),
        String::new(),
        "Executable GPU code can access your process and data.".into(),
    ];
    if !package.identity_verified() {
        lines.push("A valid signature does not verify the claimed name or source.".into());
    }
    if publisher.development {
        lines.push("Local development package; not an official published checkpoint.".into());
    } else {
        lines.push("Do not proceed if this package came from another source.".into());
    }
    lines.push("Only approve a signing key you independently trust.".into());
    if !io::stderr().is_terminal() {
        return format!("Signing key acceptance\n{}", lines.join("\n"));
    }
    let mut size: libc::winsize = unsafe { std::mem::zeroed() };
    // SAFETY: ioctl writes terminal dimensions into a valid winsize.
    unsafe { libc::ioctl(libc::STDERR_FILENO, libc::TIOCGWINSZ, &mut size) };
    let columns = if size.ws_col == 0 {
        80
    } else {
        usize::from(size.ws_col)
    };
    if columns < 28 {
        return format!("Signing key acceptance\n{}", lines.join("\n"));
    }
    let width = columns.min(80) - 4;
    let color = if package.identity_verified() {
        "36"
    } else {
        "33"
    };
    let border = format!("\x1b[{color}m");
    let title = " Signing key acceptance ";
    let mut output = format!(
        "{border}╭─{title}{}╮\x1b[0m\n",
        "─".repeat(width + 1 - title.len())
    );
    for line in lines {
        let mut remaining = line.as_str();
        while remaining.chars().count() > width {
            let end = remaining.char_indices().nth(width).unwrap().0;
            let split = remaining[..end]
                .rfind(' ')
                .filter(|n| *n > width / 2)
                .unwrap_or(end);
            let row = remaining[..split].trim_end();
            output.push_str(&format!(
                "{border}│\x1b[0m {row}{} {border}│\x1b[0m\n",
                " ".repeat(width - row.chars().count())
            ));
            remaining = remaining[split..].trim_start();
        }
        output.push_str(&format!(
            "{border}│\x1b[0m {remaining}{} {border}│\x1b[0m\n",
            " ".repeat(width - remaining.chars().count())
        ));
    }
    output.push_str(&format!("{border}╰{}╯\x1b[0m", "─".repeat(width + 2)));
    output
}

fn acceptance_line(input: impl BufRead) -> io::Result<String> {
    let mut response = String::new();
    input.take(256).read_line(&mut response)?;
    if !response.ends_with('\n') {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "publisher acceptance requires a complete line",
        ));
    }
    response.pop();
    if response.ends_with('\r') {
        response.pop();
    }
    Ok(response)
}

pub fn revoke(fingerprint: &str, given: Option<&Path>) -> Result<()> {
    if !fingerprint_valid(fingerprint) {
        return Err("a full lowercase SHA-256 fingerprint is required".into());
    }
    modify(&store_path(given)?, |store| {
        store
            .trusted
            .retain(|entry| entry.fingerprint != fingerprint)
    })?;
    eprintln!("Removed trust for signer {fingerprint}.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn acceptance_is_exact_and_requires_a_complete_bounded_line() {
        assert_eq!(
            acceptance_line(&b"local-inference-lab\n"[..]).unwrap(),
            "local-inference-lab"
        );
        assert_eq!(
            acceptance_line(&b"local-inference-lab\r\n"[..]).unwrap(),
            "local-inference-lab"
        );
        assert_eq!(
            acceptance_line(&b" local-inference-lab \n"[..]).unwrap(),
            " local-inference-lab "
        );
        assert!(acceptance_line(&b"local-inference-lab"[..]).is_err());
        assert!(acceptance_line([vec![b'x'; 256], vec![b'\n']].concat().as_slice()).is_err());
        assert!(acceptance_line(&b""[..]).is_err());
    }
}
