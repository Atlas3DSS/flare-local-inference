//! The resident agent that starts a multi-host package's peer ranks, and rank zero's client.
//!
//! Each peer host runs `flare agent` (for example as a systemd user service). Rank zero connects,
//! proves it holds the cluster's shared token, sends the executable and package the host lacks
//! into its flare cache, and asks it to run a rank; the agent streams the rank's output back and
//! stops the rank when rank zero's connection closes. The agent runs only executables that arrived
//! this way, verified against their SHA-256 names, and passes arguments without a shell.
//!
//! A host keeps only the package bytes its rank reads, at their offsets in a sparse file of the
//! package's length, and only what rank zero last sent: receiving an executable or package removes
//! the others of its kind that no connected rank zero uses.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use flare_helios::runtime_package::Publisher;
use sha2::{Digest, Sha256};

pub type Failure = Box<dyn std::error::Error + Send + Sync>;

/// The agent's default port on every host.
pub const PORT: u16 = 29900;
const PROTOCOL: &[u8] = b"flare-agent-1";
/// Sent with the token check's OK. Version 2 accepts byte ranges; version 3 also accepts
/// package directories preserving companion basenames. Version 1 sends no version byte.
const VERSION: u8 = 3;

// Frame tags.
const NONCE: u8 = 1;
const PROOF: u8 = 2;
const OK: u8 = 3;
const ERROR: u8 = 4;
const HAVE: u8 = 5;
const PUT: u8 = 6;
const RUN: u8 = 7;
const LOG: u8 = 8;
const EXIT: u8 = 9;
const FRAME_LIMIT: usize = 1 << 20;

fn write_frame(stream: &mut impl Write, tag: u8, body: &[u8]) -> std::io::Result<()> {
    let mut header = [0u8; 9];
    header[0] = tag;
    header[1..].copy_from_slice(&(body.len() as u64).to_le_bytes());
    stream.write_all(&header)?;
    stream.write_all(body)?;
    stream.flush()
}

fn read_frame(stream: &mut impl Read) -> Result<(u8, Vec<u8>), Failure> {
    let mut header = [0u8; 9];
    stream.read_exact(&mut header)?;
    let length = u64::from_le_bytes(header[1..].try_into()?) as usize;
    if length > FRAME_LIMIT {
        return Err("agent frame exceeds its limit".into());
    }
    let mut body = vec![0u8; length];
    stream.read_exact(&mut body)?;
    Ok((header[0], body))
}

fn expect(stream: &mut impl Read, tag: u8) -> Result<Vec<u8>, Failure> {
    match read_frame(stream)? {
        (found, body) if found == tag => Ok(body),
        (ERROR, body) => Err(String::from_utf8_lossy(&body).into_owned().into()),
        (found, _) => Err(format!("unexpected agent frame {found}, expected {tag}").into()),
    }
}

/// HMAC-SHA256 of `message` under `key` (RFC 2104).
fn hmac(key: &[u8], message: &[u8]) -> [u8; 32] {
    let mut block = [0u8; 64];
    if key.len() > 64 {
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

fn proof(token: &[u8], nonce: &[u8]) -> [u8; 32] {
    hmac(token, &[PROTOCOL, nonce].concat())
}

/// Read the cluster's shared token; it must be private to its owner.
pub fn read_token(path: &Path) -> Result<Vec<u8>, Failure> {
    let metadata = fs::metadata(path)
        .map_err(|e| format!("cannot read agent token {}: {e}", path.display()))?;
    if metadata.permissions().mode() & 0o077 != 0 {
        return Err(format!(
            "agent token {} must not be readable by others",
            path.display()
        )
        .into());
    }
    let token = fs::read(path)?;
    let token = token.trim_ascii();
    if token.len() < 32 {
        return Err("agent token must hold at least 32 bytes".into());
    }
    Ok(token.to_vec())
}

pub fn default_token_path() -> PathBuf {
    home().join(".config/flare/agent.token")
}

fn home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| "/".into())
}

/// Where agents keep received executables and packages.
pub fn default_cache() -> PathBuf {
    home().join(".cache/flare")
}

/// A cache entry name: one path component, no traversal.
fn entry(cache: &Path, directory: &str, name: &str) -> Result<PathBuf, Failure> {
    let components: Vec<_> = name.split('/').collect();
    if components.len() > if directory == "packages" { 2 } else { 1 }
        || components.iter().any(|part| {
            part.is_empty()
                || part.starts_with('.')
                || part.contains("..")
                || !part
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_'))
        })
    {
        return Err(format!("invalid cache entry name {name:?}").into());
    }
    Ok(cache.join(directory).join(name))
}

fn kind_directory(kind: &str) -> Result<&'static str, Failure> {
    match kind {
        "binary" => Ok("bin"),
        "package" => Ok("packages"),
        _ => Err(format!("unknown cache kind {kind:?}").into()),
    }
}

/// The SHA-256 name under which an executable is cached and verified.
pub fn binary_name(bytes: &[u8]) -> String {
    binary_name_from_digest(&Sha256::digest(bytes))
}

// ---- agent -------------------------------------------------------------------------------------

/// Serve rank zero's requests until killed.
pub fn serve(listen: &str, token: Vec<u8>, cache: PathBuf) -> Result<(), Failure> {
    serve_on(
        TcpListener::bind(listen).map_err(|e| format!("cannot listen on {listen}: {e}"))?,
        token,
        cache,
    )
}

fn serve_on(listener: TcpListener, token: Vec<u8>, cache: PathBuf) -> Result<(), Failure> {
    for directory in ["bin", "packages"] {
        fs::create_dir_all(cache.join(directory))?;
    }
    eprintln!(
        "flare agent listening on {}; cache {}",
        listener.local_addr()?,
        cache.display()
    );
    let (token, cache) = (Arc::new(token), Arc::new(cache));
    let pins = Pins::default();
    for connection in listener.incoming() {
        let Ok(stream) = connection else { continue };
        let (token, cache, pins) = (token.clone(), cache.clone(), pins.clone());
        thread::spawn(move || {
            let peer = stream
                .peer_addr()
                .map(|a| a.to_string())
                .unwrap_or_default();
            if let Err(error) = session(stream, &token, &cache, &pins) {
                eprintln!("{peer}: {error}");
            }
        });
    }
    Ok(())
}

/// Cache entries that connected rank zeros hold or run, with their connection counts.
type Pins = Arc<Mutex<HashMap<PathBuf, usize>>>;

/// One connection's pinned entries, released when the connection ends.
struct Pinned {
    pins: Pins,
    paths: Vec<PathBuf>,
}

impl Pinned {
    fn pin(&mut self, path: PathBuf) {
        *self.pins.lock().unwrap().entry(path.clone()).or_default() += 1;
        self.paths.push(path);
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        let mut pins = self.pins.lock().unwrap();
        for path in &self.paths {
            if let Some(count) = pins.get_mut(path) {
                *count -= 1;
                if *count == 0 {
                    pins.remove(path);
                }
            }
        }
    }
}

/// Remove a cache directory's entries that no connection pins, leaving transfers in progress.
fn evict(directory: &Path, pins: &Pins) -> Result<(), Failure> {
    let pins = pins.lock().unwrap();
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        let hidden = path
            .file_name()
            .is_none_or(|name| name.to_string_lossy().starts_with('.'));
        if !hidden && !pins.contains_key(&path) {
            if path.is_dir() {
                fs::remove_dir_all(&path)?;
            } else {
                fs::remove_file(&path)?;
            }
            eprintln!("removed superseded {}", path.display());
        }
    }
    Ok(())
}

fn session(mut stream: TcpStream, token: &[u8], cache: &Path, pins: &Pins) -> Result<(), Failure> {
    stream.set_nodelay(true)?;
    let mut nonce = [0u8; 32];
    File::open("/dev/urandom")?.read_exact(&mut nonce)?;
    write_frame(&mut stream, NONCE, &nonce)?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    let offered = expect(&mut stream, PROOF)?;
    let expected = proof(token, &nonce);
    // Constant-time comparison.
    if offered.len() != 32
        || offered
            .iter()
            .zip(expected)
            .fold(0, |d, (a, b)| d | (a ^ b))
            != 0
    {
        let _ = write_frame(&mut stream, ERROR, b"agent token mismatch");
        return Err("refused a client without the cluster token".into());
    }
    write_frame(&mut stream, OK, &[VERSION])?;
    stream.set_read_timeout(None)?;
    let mut pinned = Pinned {
        pins: pins.clone(),
        paths: Vec::new(),
    };
    loop {
        let (tag, body) = match read_frame(&mut stream) {
            Ok(frame) => frame,
            Err(_) => return Ok(()),
        };
        let request: serde_json::Value = serde_json::from_slice(&body)?;
        let result = match tag {
            HAVE => have(cache, &request).map(|held| {
                if let Some(path) = &held {
                    pinned.pin(cache_unit(cache, path));
                }
                vec![held.is_some() as u8]
            }),
            PUT => put(cache, &mut stream, &request).and_then(|path| {
                // The newest entry of its kind replaces those no connection uses.
                let unit = cache_unit(cache, &path);
                let directory = unit.parent().unwrap().to_owned();
                pinned.pin(unit);
                evict(&directory, pins).map(|_| Vec::new())
            }),
            RUN => return run(cache, stream, &request),
            _ => Err(format!("unknown agent request {tag}").into()),
        };
        match result {
            Ok(reply) => write_frame(&mut stream, OK, &reply)?,
            Err(error) => {
                write_frame(&mut stream, ERROR, error.to_string().as_bytes())?;
                return Err(error);
            }
        }
    }
}

fn cache_unit(cache: &Path, path: &Path) -> PathBuf {
    let packages = cache.join("packages");
    if path.parent().and_then(Path::parent) == Some(packages.as_path()) {
        path.parent().unwrap().to_owned()
    } else {
        path.to_owned()
    }
}

fn field<'a>(request: &'a serde_json::Value, name: &str) -> Result<&'a str, Failure> {
    request[name]
        .as_str()
        .ok_or_else(|| format!("agent request lacks {name}").into())
}

/// The entry's path when the cache holds it at the requested size.
fn have(cache: &Path, request: &serde_json::Value) -> Result<Option<PathBuf>, Failure> {
    let path = entry(
        cache,
        kind_directory(field(request, "kind")?)?,
        field(request, "name")?,
    )?;
    let size = request["size"].as_u64().ok_or("agent request lacks size")?;
    Ok(fs::metadata(&path)
        .is_ok_and(|m| m.len() == size)
        .then_some(path))
}

/// The byte ranges a transfer carries: the whole entry, or a package's ranges in increasing,
/// disjoint order within its size.
fn transfer_ranges(
    kind: &str,
    size: u64,
    request: &serde_json::Value,
) -> Result<Vec<(u64, u64)>, Failure> {
    let Some(listed) = request.get("ranges").filter(|r| !r.is_null()) else {
        return Ok(vec![(0, size)]);
    };
    if kind != "package" {
        return Err("only packages arrive as byte ranges".into());
    }
    let mut ranges = Vec::new();
    let mut end = 0;
    for range in listed.as_array().ok_or("ranges must be a list")? {
        let (Some(offset), Some(nbytes)) = (range[0].as_u64(), range[1].as_u64()) else {
            return Err("each range is [offset, bytes]".into());
        };
        if offset < end || nbytes == 0 || offset.checked_add(nbytes).is_none_or(|e| e > size) {
            return Err("package ranges must increase, stay disjoint and fit its size".into());
        }
        end = offset + nbytes;
        ranges.push((offset, nbytes));
    }
    Ok(ranges)
}

fn put(
    cache: &Path,
    stream: &mut TcpStream,
    request: &serde_json::Value,
) -> Result<PathBuf, Failure> {
    let kind = field(request, "kind")?;
    let name = field(request, "name")?;
    let path = entry(cache, kind_directory(kind)?, name)?;
    let size = request["size"].as_u64().ok_or("agent request lacks size")?;
    let ranges = transfer_ranges(kind, size, request)?;
    fs::create_dir_all(path.parent().unwrap())?;
    let partial = path.with_file_name(format!(
        ".{}.partial",
        path.file_name().unwrap().to_string_lossy()
    ));
    let mut output = File::create(&partial)?;
    // Bytes outside the ranges stay holes, which occupy no disk.
    output.set_len(size)?;
    let mut digest = Sha256::new();
    let mut block = vec![0u8; 16 << 20];
    for (offset, nbytes) in ranges {
        output.seek(SeekFrom::Start(offset))?;
        let mut remaining = nbytes;
        while remaining > 0 {
            let count = stream.read(&mut block[..remaining.min(16 << 20) as usize])?;
            if count == 0 {
                fs::remove_file(&partial)?;
                return Err(format!("{name}: transfer ended {remaining} bytes early").into());
            }
            if kind == "binary" {
                digest.update(&block[..count]);
            }
            output.write_all(&block[..count])?;
            remaining -= count as u64;
        }
    }
    output.sync_all()?;
    if kind == "binary" {
        let found = binary_name_from_digest(&digest.finalize());
        if found != name {
            fs::remove_file(&partial)?;
            return Err(format!("{name}: received bytes hash to {found}").into());
        }
        fs::set_permissions(&partial, fs::Permissions::from_mode(0o700))?;
    }
    fs::rename(&partial, &path)?;
    Ok(path)
}

fn binary_name_from_digest(digest: &[u8]) -> String {
    format!(
        "flare-{}",
        digest[..12]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    )
}

/// Run `bin run <package> <args>` and stream its output until it exits or the client leaves.
fn run(cache: &Path, stream: TcpStream, request: &serde_json::Value) -> Result<(), Failure> {
    let binary = entry(cache, "bin", field(request, "binary")?)?;
    let package = entry(cache, "packages", field(request, "package")?)?;
    let arguments: Vec<String> = request["arguments"]
        .as_array()
        .ok_or("agent request lacks arguments")?
        .iter()
        .map(|a| {
            a.as_str()
                .map(str::to_owned)
                .ok_or("arguments must be strings")
        })
        .collect::<Result<_, _>>()?;
    let publisher: Publisher = serde_json::from_value(request["publisher"].clone())?;
    publisher.validate().map_err(|e| e.to_string())?;
    let profile = entry(
        cache,
        "publishers",
        &format!("{}.json", publisher.fingerprint()),
    )?;
    fs::create_dir_all(profile.parent().unwrap())?;
    fs::write(&profile, serde_json::to_vec(&vec![&publisher])?)?;
    // Rank zero sends this identity only after local publisher acceptance. The peer
    // still verifies the package and matches the exact key and namespace before loading.
    let mut child = Command::new(&binary)
        .arg("run")
        .arg(&package)
        .args(&arguments)
        .arg("--publisher-keys")
        .arg(&profile)
        .arg("--dangerously-approve-key")
        .arg(publisher.fingerprint())
        .arg("--dangerously-accept-publisher")
        .arg(&publisher.namespace)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot start {}: {e}", binary.display()))?;
    let writer = Arc::new(Mutex::new(stream.try_clone()?));
    let mut forwarders = Vec::new();
    for output in [
        Box::new(child.stdout.take().unwrap()) as Box<dyn Read + Send>,
        Box::new(child.stderr.take().unwrap()),
    ] {
        let writer = writer.clone();
        forwarders.push(thread::spawn(move || {
            for line in BufReader::new(output).lines().map_while(Result::ok) {
                let mut writer = writer.lock().unwrap();
                if write_frame(&mut *writer, LOG, line.as_bytes()).is_err() {
                    break;
                }
            }
        }));
    }
    // Rank zero's connection closing stops the rank.
    let pid = child.id() as libc::pid_t;
    let mut watch = stream.try_clone()?;
    let watcher = thread::spawn(move || {
        let mut byte = [0u8; 1];
        while matches!(watch.read(&mut byte), Ok(n) if n > 0) {}
        unsafe { libc::kill(pid, libc::SIGTERM) };
    });
    let status = child.wait()?;
    for forwarder in forwarders {
        let _ = forwarder.join();
    }
    let code = status.code().unwrap_or(-1);
    let _ = write_frame(&mut *writer.lock().unwrap(), EXIT, &code.to_le_bytes());
    let _ = stream.shutdown(Shutdown::Both);
    let _ = watcher.join();
    Ok(())
}

// ---- rank zero ---------------------------------------------------------------------------------

/// A connection to one host's agent, authenticated with the cluster token.
pub struct Agent {
    host: String,
    stream: TcpStream,
    version: u8,
}

impl Agent {
    pub fn connect(host: &str, token: &[u8]) -> Result<Self, Failure> {
        let address = if host.contains(':') {
            host.to_owned()
        } else {
            format!("{host}:{PORT}")
        };
        let mut stream = TcpStream::connect(&address)
            .map_err(|e| format!("cannot reach flare agent at {address}: {e}"))?;
        stream.set_nodelay(true)?;
        let nonce = expect(&mut stream, NONCE)?;
        write_frame(&mut stream, PROOF, &proof(token, &nonce))?;
        let accepted = expect(&mut stream, OK).map_err(|e| format!("{host}: {e}"))?;
        Ok(Self {
            host: host.to_owned(),
            stream,
            version: accepted.first().copied().unwrap_or(1),
        })
    }

    /// Whether the agent stores a package as the byte ranges its rank reads.
    pub fn holds_ranges(&self) -> bool {
        self.version >= 2
    }

    pub fn holds_package_files(&self) -> bool {
        self.version >= 3
    }

    fn request(&mut self, tag: u8, request: serde_json::Value) -> Result<Vec<u8>, Failure> {
        write_frame(&mut self.stream, tag, &serde_json::to_vec(&request)?)?;
        expect(&mut self.stream, OK).map_err(|e| format!("{}: {e}", self.host).into())
    }

    /// Send `source` as cache entry `name` unless the host already holds one of the same size.
    /// With `ranges`, as from [`flare_helios::runtime_package::rank_ranges`], the host receives
    /// only those bytes of a package, at their offsets in a file of its full size.
    pub fn ensure(
        &mut self,
        kind: &str,
        name: &str,
        source: &Path,
        ranges: Option<&[(u64, u64)]>,
    ) -> Result<bool, Failure> {
        self.ensure_file(kind, name, &File::open(source)?, ranges)
    }

    pub fn ensure_file(
        &mut self,
        kind: &str,
        name: &str,
        file: &File,
        ranges: Option<&[(u64, u64)]>,
    ) -> Result<bool, Failure> {
        use std::os::unix::fs::FileExt;
        let size = file.metadata()?.len();
        let mut request = serde_json::json!({"kind": kind, "name": name, "size": size});
        if self.request(HAVE, request.clone())? == [1] {
            return Ok(false);
        }
        if let Some(ranges) = ranges {
            request["ranges"] = serde_json::json!(ranges);
        }
        write_frame(&mut self.stream, PUT, &serde_json::to_vec(&request)?)?;
        let mut block = vec![0; 16 << 20];
        for &(offset, nbytes) in ranges.unwrap_or(&[(0, size)]) {
            let mut position = 0;
            while position < nbytes {
                let count = (nbytes - position).min(block.len() as u64) as usize;
                file.read_exact_at(&mut block[..count], offset + position)?;
                self.stream.write_all(&block[..count])?;
                position += count as u64;
            }
        }
        expect(&mut self.stream, OK).map_err(|e| format!("{}: {e}", self.host))?;
        Ok(true)
    }

    /// Start `binary run <package> <arguments>` and pass each output line, then a final status
    /// line, to `output` until the rank exits. Closing the returned stream stops the rank.
    pub fn run(
        mut self,
        binary: &str,
        package: &str,
        arguments: &[String],
        publisher: &Publisher,
        mut output: impl FnMut(&str) + Send + 'static,
    ) -> Result<TcpStream, Failure> {
        let request = serde_json::json!({"binary": binary, "package": package, "arguments": arguments,
                              "publisher": publisher});
        write_frame(&mut self.stream, RUN, &serde_json::to_vec(&request)?)?;
        let keep = self.stream.try_clone()?;
        thread::spawn(move || {
            loop {
                match read_frame(&mut self.stream) {
                    Ok((LOG, line)) => output(&String::from_utf8_lossy(&line)),
                    Ok((EXIT, code)) => {
                        let code = code.try_into().map(i32::from_le_bytes).unwrap_or(-1);
                        output(&format!("exited with status {code}"));
                        break;
                    }
                    Ok((ERROR, message)) => {
                        output(&String::from_utf8_lossy(&message));
                        break;
                    }
                    _ => break,
                }
            }
        });
        Ok(keep)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmac_matches_rfc_4231_case_two() {
        let mac = hmac(b"Jefe", b"what do ya want for nothing?");
        let hex: String = mac.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            hex,
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }

    #[test]
    fn agent_admits_the_token_holder_caches_verified_files_and_runs_them() {
        let cache = std::env::temp_dir().join(format!("flare-agent-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&cache);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let token = b"0123456789abcdef0123456789abcdef".to_vec();
        {
            let (token, cache) = (token.clone(), cache.clone());
            thread::spawn(move || serve_on(listener, token, cache));
        }
        let wrong = Agent::connect(&address, b"another-token-another-token-xxxx");
        assert!(wrong.err().unwrap().to_string().contains("token mismatch"));

        let script = cache.with_extension("sh");
        fs::write(&script, "#!/bin/sh\necho \"$@\"\n").unwrap();
        let name = binary_name(&fs::read(&script).unwrap());
        let package = cache.with_extension("pkg");
        fs::write(&package, b"package bytes").unwrap();
        let mut agent = Agent::connect(&address, &token).unwrap();
        assert!(agent.ensure("binary", &name, &script, None).unwrap());
        assert!(!agent.ensure("binary", &name, &script, None).unwrap());
        assert!(
            agent
                .ensure("package", "export-1.safetensors", &package, None)
                .unwrap()
        );
        // A payload that does not hash to its name is refused.
        let mut liar = Agent::connect(&address, &token).unwrap();
        assert!(
            liar.ensure("binary", "flare-000000000000000000000000", &script, None)
                .is_err()
        );

        let (sender, receiver) = std::sync::mpsc::channel();
        let publisher = Publisher {
            name: "Test".into(),
            namespace: "test-publisher".into(),
            public_key: [1; 32],
            development: false,
        };
        let _connection = agent
            .run(
                &name,
                "export-1.safetensors",
                &["--rank".into(), "1".into()],
                &publisher,
                move |line| {
                    let _ = sender.send(line.to_owned());
                },
            )
            .unwrap();
        let first = receiver.recv_timeout(Duration::from_secs(10)).unwrap();
        assert!(
            first.starts_with("run ") && first.contains("export-1.safetensors --rank 1"),
            "{first}"
        );
        assert!(first.contains(&format!(
            "--dangerously-approve-key {}",
            publisher.fingerprint()
        )));
        assert!(first.ends_with("--dangerously-accept-publisher test-publisher"));
        let profile = entry(
            &cache,
            "publishers",
            &format!("{}.json", publisher.fingerprint()),
        )
        .unwrap();
        assert_eq!(
            serde_json::from_slice::<Vec<Publisher>>(&fs::read(profile).unwrap()).unwrap(),
            vec![publisher]
        );
        assert_eq!(
            receiver.recv_timeout(Duration::from_secs(10)).unwrap(),
            "exited with status 0"
        );
        for path in [&script, &package] {
            let _ = fs::remove_file(path);
        }
        let _ = fs::remove_dir_all(&cache);
    }

    #[test]
    fn a_host_keeps_its_rank_bytes_sparsely_and_only_the_newest_package() {
        use std::os::unix::fs::MetadataExt;

        let cache = std::env::temp_dir().join(format!("flare-agent-ranges-{}", std::process::id()));
        let _ = fs::remove_dir_all(&cache);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let token = b"0123456789abcdef0123456789abcdef".to_vec();
        {
            let (token, cache) = (token.clone(), cache.clone());
            thread::spawn(move || serve_on(listener, token, cache));
        }
        let package = cache.with_extension("pkg");
        let bytes: Vec<u8> = (0..(64u32 << 20)).map(|i| (i % 251) as u8).collect();
        fs::write(&package, &bytes).unwrap();
        let ranges = [(0, 4096), (40 << 20, 1 << 20)];
        let mut agent = Agent::connect(&address, &token).unwrap();
        assert!(agent.holds_ranges());
        assert!(
            agent
                .ensure("package", "a-rank1", &package, Some(&ranges))
                .unwrap()
        );
        let held = cache.join("packages/a-rank1");
        let copy = fs::read(&held).unwrap();
        assert_eq!(copy.len(), bytes.len());
        for &(offset, nbytes) in &ranges {
            let span = offset as usize..(offset + nbytes) as usize;
            assert_eq!(copy[span.clone()], bytes[span]);
        }
        assert!(copy[8192..1 << 20].iter().all(|&b| b == 0));
        // Each written extent may occupy a 2 MiB tmpfs hugepage; the rest stays sparse.
        assert!(fs::metadata(&held).unwrap().blocks() * 512 <= 4 << 20);
        // Ranges for an executable, or that overlap, are refused.
        let mut other = Agent::connect(&address, &token).unwrap();
        assert!(
            other
                .ensure("package", "bad", &package, Some(&[(0, 8192), (4096, 1)]))
                .is_err()
        );

        // A new package replaces those no connected rank zero uses; this connection still uses a-rank1.
        let mut second = Agent::connect(&address, &token).unwrap();
        assert!(
            second
                .ensure("package", "b-rank1", &package, Some(&ranges))
                .unwrap()
        );
        assert!(held.exists());
        drop(agent);
        let mut third = Agent::connect(&address, &token).unwrap();
        // Disconnects release their pins as the agent notices them.
        std::thread::sleep(Duration::from_millis(200));
        assert!(
            third
                .ensure("package", "c-rank1", &package, Some(&ranges))
                .unwrap()
        );
        assert!(!held.exists() && cache.join("packages/b-rank1").exists());
        let _ = fs::remove_file(&package);
        let _ = fs::remove_dir_all(&cache);
    }

    #[test]
    fn cache_entries_are_single_components() {
        assert!(entry(Path::new("/c"), "bin", "flare-00").is_ok());
        for name in ["", "../x", "a/b", ".hidden"] {
            assert!(entry(Path::new("/c"), "bin", name).is_err(), "{name}");
        }
    }
    #[test]
    fn package_directories_keep_companions_together_until_eviction() {
        let temp = tempfile::tempdir().unwrap();
        let cache = temp.path().join("cache");
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let token = b"0123456789abcdef0123456789abcdef".to_vec();
        {
            let (cache, token) = (cache.clone(), token.clone());
            thread::spawn(move || serve_on(listener, token, cache));
        }
        let source = temp.path().join("source");
        fs::write(&source, b"source bytes").unwrap();
        let mut agent = Agent::connect(&address, &token).unwrap();
        assert!(agent.holds_package_files());
        for name in ["model-data-01-of-01.safetensors", "model.safetensors"] {
            assert!(
                agent
                    .ensure(
                        "package",
                        &format!("first/{name}"),
                        &source,
                        Some(&[(0, 6)])
                    )
                    .unwrap()
            );
        }
        assert!(
            cache
                .join("packages/first/model-data-01-of-01.safetensors")
                .is_file()
        );
        let mut second = Agent::connect(&address, &token).unwrap();
        second
            .ensure("package", "second/model.safetensors", &source, None)
            .unwrap();
        assert!(cache.join("packages/first/model.safetensors").is_file());
        drop(agent);
        thread::sleep(Duration::from_millis(200));
        second
            .ensure(
                "package",
                "second/model-data-01-of-01.safetensors",
                &source,
                None,
            )
            .unwrap();
        assert!(!cache.join("packages/first").exists());
        for name in ["a/../escape", "a/b/c", "a/", "a/hidden\\name", "a/.hidden"] {
            assert!(entry(&cache, "packages", name).is_err());
        }
    }

    #[test]
    fn sharded_launch_rejects_older_agents_before_transferring_bytes() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let peer = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            write_frame(&mut stream, NONCE, &[0; 32]).unwrap();
            expect(&mut stream, PROOF).unwrap();
            write_frame(&mut stream, OK, &[2]).unwrap();
            assert_eq!(stream.read(&mut [0; 1]).unwrap(), 0);
        });
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("model.safetensors");
        fs::write(&path, b"primary").unwrap();
        let file = Arc::new(File::open(&path).unwrap());
        let shares = vec![vec![flare_helios::runtime_package::PackageFile {
            filename: "model-data-01-of-01.safetensors".into(),
            file,
            ranges: vec![(0, 7)],
        }]];
        let publisher = Publisher {
            name: "Test".into(),
            namespace: "test".into(),
            public_key: [1; 32],
            development: false,
        };
        let error = crate::launch::start(
            &[address],
            &path,
            "export",
            "127.0.0.1:29800",
            &[],
            &[0; 32],
            &publisher,
            &shares,
        )
        .err()
        .unwrap()
        .to_string();
        assert!(error.contains("protocol 3"), "{error}");
        peer.join().unwrap();
    }
}
