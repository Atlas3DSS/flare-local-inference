//! The resident agent that starts a multi-host package's peer ranks, and rank zero's client.
//!
//! Each peer host runs `flare agent` (for example as a systemd user service). Rank zero connects,
//! proves it holds the cluster's shared token, sends the executable and package the host lacks
//! into its flare cache, and asks it to run a rank; the agent streams the rank's output back and
//! stops the rank when rank zero's connection closes. The agent runs only executables that arrived
//! this way, verified against their SHA-256 names, and passes arguments without a shell.

use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use sha2::{Digest, Sha256};

pub type Failure = Box<dyn std::error::Error + Send + Sync>;

/// The agent's default port on every host.
pub const PORT: u16 = 29900;
const PROTOCOL: &[u8] = b"flare-agent-1";

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
    if name.is_empty() || name.contains('/') || name.starts_with('.') {
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
    for connection in listener.incoming() {
        let Ok(stream) = connection else { continue };
        let (token, cache) = (token.clone(), cache.clone());
        thread::spawn(move || {
            let peer = stream
                .peer_addr()
                .map(|a| a.to_string())
                .unwrap_or_default();
            if let Err(error) = session(stream, &token, &cache) {
                eprintln!("{peer}: {error}");
            }
        });
    }
    Ok(())
}

fn session(mut stream: TcpStream, token: &[u8], cache: &Path) -> Result<(), Failure> {
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
    write_frame(&mut stream, OK, &[])?;
    stream.set_read_timeout(None)?;
    loop {
        let (tag, body) = match read_frame(&mut stream) {
            Ok(frame) => frame,
            Err(_) => return Ok(()),
        };
        let request: serde_json::Value = serde_json::from_slice(&body)?;
        let result = match tag {
            HAVE => have(cache, &request).map(|present| vec![present as u8]),
            PUT => put(cache, &mut stream, &request).map(|_| Vec::new()),
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

fn field<'a>(request: &'a serde_json::Value, name: &str) -> Result<&'a str, Failure> {
    request[name]
        .as_str()
        .ok_or_else(|| format!("agent request lacks {name}").into())
}

fn have(cache: &Path, request: &serde_json::Value) -> Result<bool, Failure> {
    let path = entry(
        cache,
        kind_directory(field(request, "kind")?)?,
        field(request, "name")?,
    )?;
    let size = request["size"].as_u64().ok_or("agent request lacks size")?;
    Ok(fs::metadata(path).is_ok_and(|m| m.len() == size))
}

fn put(cache: &Path, stream: &mut TcpStream, request: &serde_json::Value) -> Result<(), Failure> {
    let kind = field(request, "kind")?;
    let name = field(request, "name")?;
    let path = entry(cache, kind_directory(kind)?, name)?;
    let size = request["size"].as_u64().ok_or("agent request lacks size")?;
    let partial = path.with_file_name(format!(".{name}.partial"));
    let mut output = File::create(&partial)?;
    let mut digest = Sha256::new();
    let mut remaining = size;
    let mut block = vec![0u8; 16 << 20];
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
    Ok(())
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
    let mut child = Command::new(&binary)
        .arg("run")
        .arg(&package)
        .args(&arguments)
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
        expect(&mut stream, OK).map_err(|e| format!("{host}: {e}"))?;
        Ok(Self {
            host: host.to_owned(),
            stream,
        })
    }

    fn request(&mut self, tag: u8, request: serde_json::Value) -> Result<Vec<u8>, Failure> {
        write_frame(&mut self.stream, tag, &serde_json::to_vec(&request)?)?;
        expect(&mut self.stream, OK).map_err(|e| format!("{}: {e}", self.host).into())
    }

    /// Send `source` as cache entry `name` unless the host already holds one of the same size.
    pub fn ensure(&mut self, kind: &str, name: &str, source: &Path) -> Result<bool, Failure> {
        let size = fs::metadata(source)?.len();
        let request = serde_json::json!({"kind": kind, "name": name, "size": size});
        if self.request(HAVE, request.clone())? == [1] {
            return Ok(false);
        }
        write_frame(&mut self.stream, PUT, &serde_json::to_vec(&request)?)?;
        std::io::copy(&mut File::open(source)?, &mut self.stream)?;
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
        mut output: impl FnMut(&str) + Send + 'static,
    ) -> Result<TcpStream, Failure> {
        let request =
            serde_json::json!({"binary": binary, "package": package, "arguments": arguments});
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
        assert!(agent.ensure("binary", &name, &script).unwrap());
        assert!(!agent.ensure("binary", &name, &script).unwrap());
        assert!(
            agent
                .ensure("package", "export-1.safetensors", &package)
                .unwrap()
        );
        // A payload that does not hash to its name is refused.
        let mut liar = Agent::connect(&address, &token).unwrap();
        assert!(
            liar.ensure("binary", "flare-000000000000000000000000", &script)
                .is_err()
        );

        let (sender, receiver) = std::sync::mpsc::channel();
        let _connection = agent
            .run(
                &name,
                "export-1.safetensors",
                &["--rank".into(), "1".into()],
                move |line| {
                    let _ = sender.send(line.to_owned());
                },
            )
            .unwrap();
        let first = receiver.recv_timeout(Duration::from_secs(10)).unwrap();
        assert!(
            first.starts_with("run ") && first.ends_with("export-1.safetensors --rank 1"),
            "{first}"
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
    fn cache_entries_are_single_components() {
        assert!(entry(Path::new("/c"), "bin", "flare-00").is_ok());
        for name in ["", "../x", "a/b", ".hidden"] {
            assert!(entry(Path::new("/c"), "bin", name).is_err(), "{name}");
        }
    }
}
