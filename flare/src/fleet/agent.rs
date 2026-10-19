//! The resident agent: one per host, started by systemd as `flare agent`.
//!
//! The agent owns its node key, its authorized controller set, and a cache of executables and
//! package files. Over an authenticated session it pairs with a controller, receives files,
//! fetches files from other nodes under a grant, runs this host's rank of a group under a grant,
//! and streams the rank's output to whoever attaches. It supervises the rank for the process's
//! life: a controller going away neither stops the rank nor loses its output, and a STOP from an
//! authorized controller sends SIGTERM.
//!
//! Request bodies are JSON except PUT, whose body is a little-endian `u32` header length, the
//! JSON header, and the transferred bytes in range order.

use std::collections::VecDeque;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::net::{TcpListener, TcpStream};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use flare_helios::runtime_package::Publisher;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::cache::{self, Pinned, Pins};
use super::grant::{Grant, Signed};
use super::identity::{Key, PublicKey};
use super::session::{FRAME_LIMIT, Role, Session};
use super::store::AgentDir;
use super::{Failure, PROTOCOL_VERSION, frame, hex, hmac_sha256, now};

/// The first frame each side sends: who it is and what it speaks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Hello {
    pub version: u32,
    /// `controller` or `node`.
    pub kind: String,
    pub name: String,
    pub public_key: String,
    /// The controller-set revision the sender holds; 0 when unpaired or for a controller.
    pub controllers_revision: u64,
    #[serde(default)]
    pub free_bytes: u64,
}

/// Exchange HELLO frames: the initiator sends first. The peer's HELLO must name the key that
/// authenticated the session and speak the same protocol.
pub fn exchange_hello(session: &mut Session, mine: &Hello) -> Result<Hello, Failure> {
    let read = |session: &mut Session| -> Result<Hello, Failure> {
        let body = session.expect(frame::HELLO)?;
        let hello: Hello = serde_json::from_slice(&body)?;
        if hello.version != PROTOCOL_VERSION {
            return Err(format!(
                "peer speaks agent protocol {}; this flare speaks {PROTOCOL_VERSION}",
                hello.version
            )
            .into());
        }
        if PublicKey::from_hex(&hello.public_key)? != session.remote() {
            return Err("peer's HELLO names a key other than the one it authenticated".into());
        }
        Ok(hello)
    };
    let body = serde_json::to_vec(mine)?;
    match session.role() {
        Role::Initiator => {
            session.write_frame(frame::HELLO, &body)?;
            read(session)
        }
        Role::Responder => {
            let theirs = read(session)?;
            session.write_frame(frame::HELLO, &body)?;
            Ok(theirs)
        }
    }
}

/// Free bytes on the filesystem holding `path`.
pub fn free_bytes(path: &Path) -> u64 {
    let Ok(path) = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()) else {
        return 0;
    };
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: path is a valid C string and stat is a valid out-pointer.
    if unsafe { libc::statvfs(path.as_ptr(), &mut stat) } != 0 {
        return 0;
    }
    stat.f_bavail as u64 * stat.f_frsize as u64
}

/// The JSON body of a frame, read within [`FRAME_LIMIT`].
fn json_body(session: &mut Session, length: u64) -> Result<Value, Failure> {
    Ok(serde_json::from_slice(
        &session.read_body(length, FRAME_LIMIT)?,
    )?)
}

fn field<'a>(request: &'a Value, name: &str) -> Result<&'a str, Failure> {
    request[name]
        .as_str()
        .ok_or_else(|| format!("request lacks {name}").into())
}

// ---- groups ------------------------------------------------------------------------------------

enum Event {
    Line(String),
    Exit(i32),
}

struct Log {
    lines: VecDeque<String>,
    bytes: usize,
    exit: Option<i32>,
    subscribers: Vec<mpsc::Sender<Event>>,
    /// Cache pins held while the rank runs.
    pins: Option<Pinned>,
}

const RETAINED_BYTES: usize = 64 << 10;

/// A rank process this agent started and supervises.
pub struct Group {
    pub id: String,
    pub rank: usize,
    pub package: String,
    pub pid: i32,
    started: Instant,
    log: Mutex<Log>,
}

impl Group {
    fn push(&self, line: String) {
        let mut log = self.log.lock().unwrap();
        log.bytes += line.len();
        log.lines.push_back(line.clone());
        while log.bytes > RETAINED_BYTES && log.lines.len() > 1 {
            let dropped = log.lines.pop_front().unwrap();
            log.bytes -= dropped.len();
        }
        log.subscribers
            .retain(|s| s.send(Event::Line(line.clone())).is_ok());
    }

    fn finish(&self, code: i32) {
        let mut log = self.log.lock().unwrap();
        log.exit = Some(code);
        log.subscribers
            .retain(|s| s.send(Event::Exit(code)).is_ok());
        log.subscribers.clear();
        log.pins = None;
    }

    pub fn running(&self) -> bool {
        self.log.lock().unwrap().exit.is_none()
    }

    pub fn exit(&self) -> Option<i32> {
        self.log.lock().unwrap().exit
    }

    /// The retained output, the exit status if finished, and a receiver of later events.
    fn subscribe(&self) -> (Vec<String>, Option<i32>, mpsc::Receiver<Event>) {
        let mut log = self.log.lock().unwrap();
        let (sender, receiver) = mpsc::channel();
        if log.exit.is_none() {
            log.subscribers.push(sender);
        }
        (log.lines.iter().cloned().collect(), log.exit, receiver)
    }
}

// ---- state -------------------------------------------------------------------------------------

pub struct Agent {
    pub dir: AgentDir,
    pub key: Key,
    pub cache: PathBuf,
    pub name: String,
    pins: Pins,
    group: Mutex<Option<Arc<Group>>>,
}

impl Agent {
    pub fn new(dir: AgentDir, cache: PathBuf) -> Result<Self, Failure> {
        let key = dir.key()?;
        for directory in ["bin", "packages", "publishers"] {
            fs::create_dir_all(cache.join(directory))?;
        }
        Ok(Self {
            dir,
            key,
            cache,
            name: super::hostname(),
            pins: Pins::default(),
            group: Mutex::new(None),
        })
    }

    pub fn hello(&self) -> Result<Hello, Failure> {
        Ok(Hello {
            version: PROTOCOL_VERSION,
            kind: "node".into(),
            name: self.name.clone(),
            public_key: self.key.public().to_hex(),
            controllers_revision: self.dir.controllers()?.map_or(0, |(_, set)| set.revision),
            free_bytes: free_bytes(&self.cache),
        })
    }

    fn is_controller(&self, key: &PublicKey) -> Result<bool, Failure> {
        Ok(self
            .dir
            .controllers()?
            .is_some_and(|(_, set)| set.contains(key)))
    }

    /// Verify a grant against this node's controllers.
    fn grant(&self, signed: &Value) -> Result<Grant, Failure> {
        let signed: Signed = serde_json::from_value(signed.clone())?;
        let Some((_, set)) = self.dir.controllers()? else {
            return Err("this node is not paired with any controller".into());
        };
        let grant: Grant = signed.verify(|k| set.contains(k))?;
        if grant.members.is_empty() || grant.package.files.is_empty() {
            return Err("grant names no members or files".into());
        }
        Ok(grant)
    }

    fn current_group(&self) -> Option<Arc<Group>> {
        self.group.lock().unwrap().clone()
    }
}

/// Serve sessions on `listen` until killed.
pub fn serve(listen: &str, dir: AgentDir, cache: PathBuf) -> Result<(), Failure> {
    let listener =
        TcpListener::bind(listen).map_err(|e| format!("cannot listen on {listen}: {e}"))?;
    let agent = Arc::new(Agent::new(dir, cache)?);
    eprintln!(
        "flare agent {} ({}) listening on {}; cache {}; {}",
        agent.name,
        agent.key.public().fingerprint(),
        listener.local_addr()?,
        agent.cache.display(),
        match agent.dir.controllers()? {
            Some((_, set)) => format!("controller set revision {}", set.revision),
            None => "unpaired".to_owned(),
        }
    );
    serve_on(listener, agent)
}

pub fn serve_on(listener: TcpListener, agent: Arc<Agent>) -> Result<(), Failure> {
    for connection in listener.incoming() {
        let Ok(stream) = connection else { continue };
        let agent = agent.clone();
        thread::spawn(move || {
            let peer = stream
                .peer_addr()
                .map(|a| a.to_string())
                .unwrap_or_default();
            if let Err(error) = session(&agent, stream) {
                eprintln!("{peer}: {error}");
            }
        });
    }
    Ok(())
}

fn session(agent: &Arc<Agent>, stream: TcpStream) -> Result<(), Failure> {
    let mut session = Session::accept(stream, &agent.key)?;
    let hello = exchange_hello(&mut session, &agent.hello()?)?;
    let mut pinned = Pinned::new(&agent.pins);
    loop {
        let (tag, length) = match session.read_header() {
            Ok(header) => header,
            Err(_) => return Ok(()),
        };
        let authorized = agent.is_controller(&session.remote())?;
        let result = match tag {
            frame::PAIR => {
                let request = json_body(&mut session, length)?;
                pair(agent, &mut session, &hello, &request)
            }
            frame::FETCH => {
                let request = json_body(&mut session, length)?;
                match fetch(agent, &mut session, &request) {
                    Ok(()) => continue,
                    Err(error) => Err(error),
                }
            }
            _ if !authorized => Err(format!(
                "{} is not an authorized controller of this node",
                session.remote().fingerprint()
            )
            .into()),
            frame::CONTROLLERS => {
                let record: Signed = serde_json::from_value(json_body(&mut session, length)?)?;
                agent
                    .dir
                    .apply_controllers(&record)
                    .map(|set| json!({"revision": set.revision}).to_string().into_bytes())
            }
            frame::HAVE => {
                let request = json_body(&mut session, length)?;
                have(agent, &mut pinned, &request)
            }
            frame::PUT => put(agent, &mut session, &mut pinned, length),
            frame::TRANSFER => {
                let request = json_body(&mut session, length)?;
                transfer(agent, &mut pinned, &request)
            }
            frame::RUN | frame::ATTACH => {
                let request = json_body(&mut session, length)?;
                let started = if tag == frame::RUN {
                    run(agent, &request)
                } else {
                    attach(agent, &request)
                };
                match started {
                    Ok(group) => {
                        session.ok(json!({"pid": group.pid, "group": group.id})
                            .to_string()
                            .as_bytes())?;
                        return stream_group(&mut session, &group);
                    }
                    Err(error) => Err(error),
                }
            }
            frame::STOP => {
                let request = json_body(&mut session, length)?;
                stop(agent, &request)
            }
            frame::STATUS => {
                let request = json_body(&mut session, length)?;
                status(agent, &request)
            }
            frame::GET => {
                let request = json_body(&mut session, length)?;
                match get(&mut session, &request) {
                    Ok(()) => continue,
                    Err(error) => Err(error),
                }
            }
            frame::STAT => {
                let request = json_body(&mut session, length)?;
                stat(&request)
            }
            _ => Err(format!("unknown request {tag}").into()),
        };
        match result {
            Ok(reply) => session.ok(&reply)?,
            Err(error) => {
                session.error(&error.to_string())?;
                return Err(error);
            }
        }
    }
}

fn constant_time_equal(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0, |d, (x, y)| d | (x ^ y)) == 0
}

fn pair(
    agent: &Agent,
    session: &mut Session,
    hello: &Hello,
    request: &Value,
) -> Result<Vec<u8>, Failure> {
    if agent.dir.controllers()?.is_some() {
        return Err("this node is paired; add controllers with `flare fleet controllers add` or run `flare agent reset` on it".into());
    }
    let record: Signed = serde_json::from_value(request["controllers"].clone())
        .map_err(|_| "pairing needs the controller-set record")?;
    if record.signer()? != session.remote() {
        return Err("the controller-set record must be signed by the pairing controller".into());
    }
    let mut reply = json!({
        "name": agent.name,
        "public_key": agent.key.public().to_hex(),
        "fingerprint": agent.key.public().fingerprint(),
    });
    let hash = *session.handshake_hash();
    match (agent.dir.join_code()?, request["proof"].as_str()) {
        (Some(code), Some(proof)) => {
            let expected = hmac_sha256(&code.key, &[b"controller".as_slice(), &hash].concat());
            if !constant_time_equal(&super::unhex(proof)?, &expected) {
                return Err("join code proof rejected".into());
            }
            let set = agent.dir.apply_controllers(&record)?;
            agent.dir.consume_join_code()?;
            agent.dir.clear_accept_first();
            reply["proof"] = hex(&hmac_sha256(
                &code.key,
                &[b"node".as_slice(), &hash].concat(),
            ))
            .into();
            reply["revision"] = set.revision.into();
        }
        _ if agent.dir.accept_first() => {
            let set = agent.dir.apply_controllers(&record)?;
            agent.dir.clear_accept_first();
            agent.dir.consume_join_code()?;
            reply["revision"] = set.revision.into();
        }
        (Some(_), None) => return Err("this node requires its join code".into()),
        (None, _) => {
            return Err(
                "this node accepts no pairing: install it with a join code or --accept-first"
                    .into(),
            );
        }
    }
    eprintln!(
        "paired with controller {} ({})",
        hello.name,
        session.remote().fingerprint()
    );
    Ok(reply.to_string().into_bytes())
}

fn have(agent: &Agent, pinned: &mut Pinned, request: &Value) -> Result<Vec<u8>, Failure> {
    let size = request["size"].as_u64().ok_or("request lacks size")?;
    let held = cache::have(
        &agent.cache,
        field(request, "kind")?,
        field(request, "name")?,
        size,
    )?;
    if let Some(path) = &held {
        pinned.pin(cache::unit(&agent.cache, path));
    }
    Ok(vec![held.is_some() as u8])
}

fn put(
    agent: &Agent,
    session: &mut Session,
    pinned: &mut Pinned,
    length: u64,
) -> Result<Vec<u8>, Failure> {
    let mut body = session.body();
    let mut prefix = [0u8; 4];
    body.read_exact(&mut prefix)?;
    let header_length = u32::from_le_bytes(prefix) as usize;
    if header_length as u64 + 4 > length || header_length > FRAME_LIMIT as usize {
        return Err("invalid PUT header".into());
    }
    let mut header = vec![0u8; header_length];
    body.read_exact(&mut header)?;
    let request: Value = serde_json::from_slice(&header)?;
    let kind = field(&request, "kind")?;
    let name = field(&request, "name")?;
    let size = request["size"].as_u64().ok_or("request lacks size")?;
    let ranges = cache::ranges_from_json(size, request.get("ranges"))?;
    if cache::range_bytes(&ranges) + 4 + header_length as u64 != length {
        return Err("PUT length does not match its ranges".into());
    }
    let path = cache::store(&agent.cache, kind, name, size, &ranges, &mut body)?;
    let unit = cache::unit(&agent.cache, &path);
    pinned.pin(unit.clone());
    for removed in cache::evict(unit.parent().unwrap(), &agent.pins)? {
        eprintln!("removed superseded {}", removed.display());
    }
    Ok(Vec::new())
}

/// Fetch the files of a package from another node under a grant into the cache.
fn transfer(agent: &Agent, pinned: &mut Pinned, request: &Value) -> Result<Vec<u8>, Failure> {
    agent.grant(&request["grant"])?;
    let source_key = PublicKey::from_hex(field(&request["source"], "public_key")?)?;
    let address = field(&request["source"], "address")?;
    let directory = field(request, "directory")?;
    let files = request["files"].as_array().ok_or("request lacks files")?;
    let mut session = Session::connect(address, &agent.key)?;
    let theirs = exchange_hello(&mut session, &agent.hello()?)?;
    if session.remote() != source_key {
        return Err(format!(
            "{address} is {} ({}), not the grant's source node",
            theirs.name,
            session.remote().fingerprint()
        )
        .into());
    }
    let mut received = 0u64;
    for file in files {
        let name = field(file, "name")?;
        let size = file["size"].as_u64().ok_or("file lacks size")?;
        let ranges = cache::ranges_from_json(size, file.get("ranges"))?;
        let entry = format!("{directory}/{name}");
        if let Some(path) = cache::have(&agent.cache, "package", &entry, size)? {
            pinned.pin(cache::unit(&agent.cache, &path));
            continue;
        }
        session.write_frame(
            frame::FETCH,
            &json!({"grant": request["grant"], "name": name, "ranges": ranges})
                .to_string()
                .into_bytes(),
        )?;
        let (tag, length) = session.read_header()?;
        if tag == frame::ERROR {
            return Err(
                String::from_utf8_lossy(&session.read_body(length, FRAME_LIMIT)?)
                    .into_owned()
                    .into(),
            );
        }
        if tag != frame::OK || length != cache::range_bytes(&ranges) {
            return Err(format!("{address}: unexpected reply to FETCH {name}").into());
        }
        let path = cache::store(
            &agent.cache,
            "package",
            &entry,
            size,
            &ranges,
            &mut session.body(),
        )?;
        pinned.pin(cache::unit(&agent.cache, &path));
        received += length;
    }
    for removed in cache::evict(&agent.cache.join("packages"), &agent.pins)? {
        eprintln!("removed superseded {}", removed.display());
    }
    Ok(json!({"received": received}).to_string().into_bytes())
}

/// Serve byte ranges of a package file to a member of the grant.
fn fetch(agent: &Agent, session: &mut Session, request: &Value) -> Result<(), Failure> {
    let grant = agent.grant(&request["grant"])?;
    if grant.member(&session.remote()).is_none() {
        return Err(format!(
            "{} is not a member of the grant",
            session.remote().fingerprint()
        )
        .into());
    }
    let name = field(request, "name")?;
    if !grant.package.files.iter().any(|f| f == name) {
        return Err(format!("{name} is not a file of the grant's package").into());
    }
    let mine = agent.key.public().to_hex();
    let path = match &grant.package.source {
        Some(source) if source.public_key == mine => Path::new(&source.path)
            .parent()
            .ok_or("source path has no directory")?
            .join(name),
        _ => {
            let complete = cache::entry(
                &agent.cache,
                "packages",
                &format!("{}/{name}", grant.complete_directory()),
            )?;
            if complete.is_file() {
                complete
            } else {
                let rank = grant
                    .member(&agent.key.public())
                    .map(|m| m.rank)
                    .ok_or("this node holds no copy of the package")?;
                cache::entry(
                    &agent.cache,
                    "packages",
                    &format!("{}/{name}", grant.cache_directory(rank)),
                )?
            }
        }
    };
    let file = File::open(&path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let size = file.metadata()?.len();
    let ranges = cache::ranges_from_json(size, request.get("ranges"))?;
    let mut reader = cache::RangeReader::new(&file, &ranges);
    session.write_frame_from(frame::OK, cache::range_bytes(&ranges), &mut reader)
}

/// Read a range of a node-local file: how a controller mirrors a package's header and index.
fn get(session: &mut Session, request: &Value) -> Result<(), Failure> {
    let path = Path::new(field(request, "path")?);
    if !path.is_absolute() {
        return Err("GET needs an absolute path".into());
    }
    let offset = request["offset"].as_u64().ok_or("request lacks offset")?;
    let nbytes = request["nbytes"].as_u64().ok_or("request lacks nbytes")?;
    if nbytes > 256 << 20 {
        return Err("GET reads at most 256 MiB".into());
    }
    let file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let size = file.metadata()?.len();
    let ranges = [(offset, nbytes)];
    cache::validate_ranges(size, &ranges)?;
    let mut reader = cache::RangeReader::new(&file, &ranges);
    session.write_frame_from(frame::OK, nbytes, &mut reader)
}

/// A package's primary file size and its companions with their sizes.
fn stat(request: &Value) -> Result<Vec<u8>, Failure> {
    let path = Path::new(field(request, "path")?);
    if !path.is_absolute() {
        return Err("STAT needs an absolute path".into());
    }
    let size = fs::metadata(path)
        .map_err(|e| format!("cannot stat {}: {e}", path.display()))?
        .len();
    let directory = path.parent().ok_or("path has no directory")?;
    let companions = flare_helios::runtime_package::package_companions(path)
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|name| {
            let size = fs::metadata(directory.join(&name))
                .map_err(|e| format!("cannot stat companion {name}: {e}"))?
                .len();
            Ok(json!({"name": name, "size": size}))
        })
        .collect::<Result<Vec<_>, Failure>>()?;
    Ok(json!({"size": size, "companions": companions})
        .to_string()
        .into_bytes())
}

fn run(agent: &Arc<Agent>, request: &Value) -> Result<Arc<Group>, Failure> {
    let grant = agent.grant(&request["grant"])?;
    let rank = request["rank"].as_u64().ok_or("request lacks rank")? as usize;
    let mine = agent.key.public();
    let member = grant
        .rank(rank)
        .ok_or_else(|| format!("the grant has no rank {rank}"))?;
    if member.public_key != mine.to_hex() {
        return Err(format!("the grant names another node at rank {rank}").into());
    }
    if !grant.current(now()) {
        return Err("the grant has expired".into());
    }
    let binary_name = field(request, "binary")?;
    let binary = cache::entry(&agent.cache, "bin", binary_name)?;
    if !binary.is_file() {
        return Err(format!("executable {binary_name} is not cached").into());
    }
    let publisher: Publisher = serde_json::from_value(request["publisher"].clone())?;
    publisher.validate().map_err(|e| e.to_string())?;
    let primary = grant.primary();
    let mut pins = Pinned::new(&agent.pins);
    pins.pin(binary.clone());
    let (package, fetched) = match &grant.package.source {
        Some(source) if source.public_key == mine.to_hex() => (PathBuf::from(&source.path), false),
        _ => {
            let directory = if rank == 0 {
                grant.complete_directory()
            } else {
                grant.cache_directory(rank)
            };
            let path = cache::entry(&agent.cache, "packages", &format!("{directory}/{primary}"))?;
            pins.pin(cache::unit(&agent.cache, &path));
            // Rank zero's complete copy came from the controller; a peer's from another node.
            (path, rank != 0)
        }
    };
    if !package.is_file() {
        return Err(format!("package {} is not on this node", package.display()).into());
    }
    let port = grant
        .rendezvous
        .rsplit_once(':')
        .map(|(_, port)| port)
        .ok_or("grant rendezvous lacks a port")?;
    let profile = agent
        .cache
        .join("publishers")
        .join(format!("{}.json", publisher.fingerprint()));
    fs::write(&profile, serde_json::to_vec(&vec![&publisher])?)?;
    let mut command = Command::new(&binary);
    command
        .arg("run")
        .arg(&package)
        .arg("--rank")
        .arg(rank.to_string())
        .arg("--rendezvous")
        .arg(if rank == 0 {
            format!("0.0.0.0:{port}")
        } else {
            grant.rendezvous.clone()
        })
        .args(&grant.serving);
    if rank == 0 {
        command.args(&grant.rank_zero);
    }
    if grant.roce_ring {
        command.arg("--roce-ring");
    }
    // Bytes fetched from another node are verified before loading.
    if fetched && !grant.serving.iter().any(|a| a == "--verify-weights") {
        command.arg("--verify-weights");
    }
    // The controller accepted the publisher at its own terminal; the rank still verifies the
    // package and matches the exact key and namespace before loading.
    command
        .arg("--publisher-keys")
        .arg(&profile)
        .arg("--dangerously-approve-key")
        .arg(publisher.fingerprint())
        .arg("--dangerously-accept-publisher")
        .arg(&publisher.namespace)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut slot = agent.group.lock().unwrap();
    if let Some(current) = slot.as_ref().filter(|g| g.running()) {
        return Err(format!(
            "group {} is running on this node as rank {}",
            current.id, current.rank
        )
        .into());
    }
    let mut child = command
        .spawn()
        .map_err(|e| format!("cannot start {}: {e}", binary.display()))?;
    let group = Arc::new(Group {
        id: grant.group.clone(),
        rank,
        package: package.display().to_string(),
        pid: child.id() as i32,
        started: Instant::now(),
        log: Mutex::new(Log {
            lines: VecDeque::new(),
            bytes: 0,
            exit: None,
            subscribers: Vec::new(),
            pins: Some(pins),
        }),
    });
    *slot = Some(group.clone());
    drop(slot);
    eprintln!("group {} rank {rank}: started pid {}", group.id, group.pid);
    let mut forwarders = Vec::new();
    for output in [
        Box::new(child.stdout.take().unwrap()) as Box<dyn Read + Send>,
        Box::new(child.stderr.take().unwrap()),
    ] {
        let group = group.clone();
        forwarders.push(thread::spawn(move || {
            for line in BufReader::new(output).lines().map_while(Result::ok) {
                group.push(line);
            }
        }));
    }
    let finished = group.clone();
    thread::spawn(move || {
        let status = child.wait();
        for forwarder in forwarders {
            let _ = forwarder.join();
        }
        let code = status.ok().and_then(|s| s.code()).unwrap_or(-1);
        eprintln!(
            "group {} rank {}: exited with status {code}",
            finished.id, finished.rank
        );
        finished.finish(code);
    });
    Ok(group)
}

fn named_group(agent: &Agent, request: &Value) -> Result<Arc<Group>, Failure> {
    let group = agent
        .current_group()
        .ok_or("no group has run on this node")?;
    if let Some(wanted) = request["group"].as_str() {
        if wanted != group.id {
            return Err(format!("group {wanted} is not on this node; {} is", group.id).into());
        }
    }
    Ok(group)
}

fn attach(agent: &Agent, request: &Value) -> Result<Arc<Group>, Failure> {
    named_group(agent, request)
}

/// Send the group's retained output, then live output until it exits or the peer leaves.
fn stream_group(session: &mut Session, group: &Group) -> Result<(), Failure> {
    let (retained, exit, events) = group.subscribe();
    for line in retained {
        session.write_frame(frame::LOG, line.as_bytes())?;
    }
    if let Some(code) = exit {
        return session.write_frame(frame::EXIT, &code.to_le_bytes());
    }
    loop {
        match events.recv_timeout(Duration::from_secs(20)) {
            Ok(Event::Line(line)) => session.write_frame(frame::LOG, line.as_bytes())?,
            Ok(Event::Exit(code)) => return session.write_frame(frame::EXIT, &code.to_le_bytes()),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // An empty LOG frame keeps a silent attachment alive and detects a departed peer.
                session.write_frame(frame::LOG, b"")?;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let code = group.exit().unwrap_or(-1);
                return session.write_frame(frame::EXIT, &code.to_le_bytes());
            }
        }
    }
}

fn stop(agent: &Agent, request: &Value) -> Result<Vec<u8>, Failure> {
    let group = named_group(agent, request)?;
    let running = group.running();
    if running {
        // SAFETY: pid is this agent's child.
        unsafe { libc::kill(group.pid, libc::SIGTERM) };
        eprintln!("group {} rank {}: stop requested", group.id, group.rank);
    }
    Ok(json!({"stopped": running}).to_string().into_bytes())
}

fn status(agent: &Agent, request: &Value) -> Result<Vec<u8>, Failure> {
    let group = agent.current_group().map(|g| {
        json!({
            "id": g.id,
            "rank": g.rank,
            "package": g.package,
            "pid": g.pid,
            "running": g.running(),
            "exit": g.exit(),
            "uptime_seconds": g.started.elapsed().as_secs(),
        })
    });
    let mut reply = json!({
        "name": agent.name,
        "public_key": agent.key.public().to_hex(),
        "fingerprint": agent.key.public().fingerprint(),
        "version": PROTOCOL_VERSION,
        "free_bytes": free_bytes(&agent.cache),
        "controllers_revision": agent.dir.controllers()?.map_or(0, |(_, s)| s.revision),
        "group": group,
    });
    if let Some(toward) = request["toward"].as_str() {
        reply["address_toward"] = super::address_toward(toward)?.to_string().into();
    }
    Ok(reply.to_string().into_bytes())
}

// ---- install -----------------------------------------------------------------------------------

/// Install this executable as the host's agent service and prepare pairing.
pub fn install(listen: &str, join_code: Option<&str>, accept_first: bool) -> Result<(), Failure> {
    let dir = AgentDir::new(AgentDir::default_root());
    let key = dir.key()?;
    let home = super::home();
    let destination = home.join(".local/lib/flare-agent/flare");
    fs::create_dir_all(destination.parent().unwrap())?;
    let source = std::env::current_exe()?;
    let temporary = destination.with_extension("tmp");
    fs::copy(&source, &temporary)?;
    fs::set_permissions(&temporary, fs::Permissions::from_mode(0o755))?;
    fs::rename(&temporary, &destination)?;
    let unit_dir = home.join(".config/systemd/user");
    fs::create_dir_all(&unit_dir)?;
    fs::write(
        unit_dir.join("flare-agent.service"),
        format!(
            "[Unit]\nDescription=Flare agent: runs this host's rank of a fleet's serving group\nAfter=network-online.target\n\n[Service]\nExecStart=%h/.local/lib/flare-agent/flare agent --listen {listen}\nRestart=on-failure\nRestartSec=5\n\n[Install]\nWantedBy=default.target\n"
        ),
    )?;
    let code = if accept_first {
        dir.set_accept_first()?;
        None
    } else {
        let code = match join_code {
            Some(code) => super::normalize_code(code),
            None => super::store::new_join_code()?,
        };
        if code.len() < 16 {
            return Err("a join code needs at least 16 characters".into());
        }
        dir.set_join_code(&code, 24 * 3600)?;
        Some(code)
    };
    let mut service = Vec::new();
    for arguments in [
        ["systemctl", "--user", "daemon-reload"].as_slice(),
        &["systemctl", "--user", "enable", "flare-agent"],
        &["systemctl", "--user", "restart", "flare-agent"],
    ] {
        match Command::new(arguments[0]).args(&arguments[1..]).status() {
            Ok(status) if status.success() => {}
            Ok(status) => service.push(format!("{} exited with {status}", arguments.join(" "))),
            Err(error) => service.push(format!("{}: {error}", arguments.join(" "))),
        }
    }
    if let Ok(user) = std::env::var("USER") {
        let _ = Command::new("loginctl")
            .args(["enable-linger", &user])
            .status();
    }
    let name = super::hostname();
    println!("flare agent installed at {}", destination.display());
    println!("node name:        {name}");
    println!("node fingerprint: {}", key.public().fingerprint());
    println!("listening on:     {listen}");
    match code {
        Some(code) => println!(
            "join code:        {code}\n                  (valid 24 hours; from the controller: flare fleet pair {name} --code {code})"
        ),
        None => println!("pairing:          accepts the first controller that connects"),
    }
    if service.is_empty() {
        println!(
            "service:          flare-agent enabled and running (journalctl --user -u flare-agent)"
        );
        Ok(())
    } else {
        Err(format!(
            "agent files are installed, but the service could not be started: {}",
            service.join("; ")
        )
        .into())
    }
}

pub fn reset() -> Result<(), Failure> {
    let dir = AgentDir::new(AgentDir::default_root());
    dir.reset()?;
    println!(
        "node {} is unpaired; install the agent again to accept a controller",
        super::hostname()
    );
    Ok(())
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::fleet::grant::{ControllerKey, ControllerSet, Member, Package, Source};
    use std::net::TcpListener;

    /// An agent on a free port in a temporary directory, with its controller's key.
    pub struct Harness {
        pub address: String,
        pub agent: Arc<Agent>,
        pub controller: Key,
        pub temp: tempfile::TempDir,
    }

    pub fn harness(seed: u8) -> Harness {
        let temp = tempfile::tempdir().unwrap();
        let dir = AgentDir::new(temp.path().join("agent"));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let agent = Arc::new(Agent::new(dir, temp.path().join("cache")).unwrap());
        {
            let agent = agent.clone();
            thread::spawn(move || serve_on(listener, agent));
        }
        Harness {
            address,
            agent,
            controller: Key::from_seed(&[seed; 32]).unwrap(),
            temp,
        }
    }

    pub fn controller_hello(key: &Key) -> Hello {
        Hello {
            version: PROTOCOL_VERSION,
            kind: "controller".into(),
            name: "test-controller".into(),
            public_key: key.public().to_hex(),
            controllers_revision: 0,
            free_bytes: 0,
        }
    }

    pub fn connect(harness: &Harness, key: &Key) -> Session {
        let mut session = Session::connect(&harness.address, key).unwrap();
        exchange_hello(&mut session, &controller_hello(key)).unwrap();
        session
    }

    pub fn record(key: &Key) -> Signed {
        Signed::sign(
            key,
            &ControllerSet {
                revision: 1,
                keys: vec![ControllerKey {
                    name: "test-controller".into(),
                    public_key: key.public().to_hex(),
                }],
            },
        )
        .unwrap()
    }

    /// Pair through accept-first and return an authorized session.
    pub fn pair_accept_first(harness: &Harness) -> Session {
        harness.agent.dir.set_accept_first().unwrap();
        let mut session = connect(harness, &harness.controller);
        session
            .request(
                frame::PAIR,
                &json!({"controllers": record(&harness.controller)}),
            )
            .unwrap();
        session
    }

    pub fn put(
        session: &mut Session,
        kind: &str,
        name: &str,
        bytes: &[u8],
        ranges: Option<&[(u64, u64)]>,
    ) {
        let header =
            json!({"kind": kind, "name": name, "size": bytes.len(), "ranges": ranges}).to_string();
        let mut body = (header.len() as u32).to_le_bytes().to_vec();
        body.extend(header.as_bytes());
        for &(offset, nbytes) in ranges.unwrap_or(&[(0, bytes.len() as u64)]) {
            body.extend(&bytes[offset as usize..(offset + nbytes) as usize]);
        }
        session.write_frame(frame::PUT, &body).unwrap();
        session.expect(frame::OK).unwrap();
    }

    fn grant_for(harness: &Harness, rank: usize, source: Option<Source>) -> Signed {
        Signed::sign(
            &harness.controller,
            &Grant {
                group: "test-group".into(),
                package: Package {
                    export: "export".into(),
                    length: 13,
                    files: vec!["model.safetensors".into()],
                    source,
                },
                members: vec![Member {
                    rank,
                    name: harness.agent.name.clone(),
                    public_key: harness.agent.key.public().to_hex(),
                    address: harness.address.clone(),
                }],
                rendezvous: "127.0.0.1:29800".into(),
                serving: vec!["--max-concurrency".into(), "2".into()],
                rank_zero: vec!["--port".into(), "8001".into()],
                roce_ring: false,
                issued: now(),
                expires: now() + 60,
            },
        )
        .unwrap()
    }

    fn publisher() -> Publisher {
        Publisher {
            name: "Test".into(),
            namespace: "test-publisher".into(),
            public_key: [1; 32],
            development: false,
        }
    }

    fn run_request(grant: &Signed, rank: usize, binary: &str) -> Value {
        json!({"grant": grant, "rank": rank, "binary": binary, "publisher": publisher()})
    }

    fn collect(session: &mut Session) -> (Vec<String>, i32) {
        let mut lines = Vec::new();
        loop {
            match session.read_frame().unwrap() {
                (frame::LOG, line) if line.is_empty() => {}
                (frame::LOG, line) => lines.push(String::from_utf8(line).unwrap()),
                (frame::EXIT, code) => {
                    return (lines, i32::from_le_bytes(code.try_into().unwrap()));
                }
                (tag, body) => panic!("unexpected frame {tag}: {}", String::from_utf8_lossy(&body)),
            }
        }
    }

    #[test]
    fn unpaired_nodes_refuse_requests_and_pair_through_their_join_code() {
        let harness = harness(1);
        let mut session = connect(&harness, &harness.controller);
        let refused = session
            .request(frame::STATUS, &json!({}))
            .err()
            .unwrap()
            .to_string();
        assert!(
            refused.contains("not an authorized controller"),
            "{refused}"
        );
        let mut session = connect(&harness, &harness.controller);
        let refused = session
            .request(
                frame::PAIR,
                &json!({"controllers": record(&harness.controller)}),
            )
            .err()
            .unwrap()
            .to_string();
        assert!(refused.contains("accepts no pairing"), "{refused}");
        let mut session = connect(&harness, &harness.controller);
        session
            .write_frame(
                frame::CONTROLLERS,
                &serde_json::to_vec(&record(&harness.controller)).unwrap(),
            )
            .unwrap();
        assert!(
            session
                .expect(frame::OK)
                .err()
                .unwrap()
                .to_string()
                .contains("not an authorized")
        );

        harness
            .agent
            .dir
            .set_join_code("ABCD-EFGH-IJKL-MNOP", 60)
            .unwrap();
        let mut session = connect(&harness, &harness.controller);
        let wrong = hmac_sha256(&[0; 32], session.handshake_hash());
        let refused = session
            .request(
                frame::PAIR,
                &json!({"proof": hex(&wrong), "controllers": record(&harness.controller)}),
            )
            .err()
            .unwrap()
            .to_string();
        assert!(refused.contains("proof rejected"), "{refused}");
        let other = Key::from_seed(&[9; 32]).unwrap();
        let mut session = connect(&harness, &harness.controller);
        assert!(
            session
                .request(frame::PAIR, &json!({"controllers": record(&other)}))
                .is_err()
        );

        let mut session = connect(&harness, &harness.controller);
        let key = crate::fleet::store::join_key("abcd efgh ijkl mnop");
        let proof = hmac_sha256(
            &key,
            &[b"controller".as_slice(), session.handshake_hash()].concat(),
        );
        let reply: Value = serde_json::from_slice(
            &session
                .request(
                    frame::PAIR,
                    &json!({"proof": hex(&proof), "controllers": record(&harness.controller)}),
                )
                .unwrap(),
        )
        .unwrap();
        let expected = hmac_sha256(
            &key,
            &[b"node".as_slice(), session.handshake_hash()].concat(),
        );
        assert_eq!(reply["proof"], hex(&expected));
        assert_eq!(reply["public_key"], harness.agent.key.public().to_hex());
        assert!(harness.agent.dir.join_code().unwrap().is_none());

        let status: Value = serde_json::from_slice(
            &session
                .request(frame::STATUS, &json!({"toward": "127.0.0.1"}))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(status["controllers_revision"], 1);
        assert_eq!(status["address_toward"], "127.0.0.1");
        assert!(status["group"].is_null());
        let mut stranger = connect(&harness, &other);
        assert!(stranger.request(frame::STATUS, &json!({})).is_err());
        let mut again = connect(&harness, &harness.controller);
        let refused = again
            .request(
                frame::PAIR,
                &json!({"controllers": record(&harness.controller)}),
            )
            .err()
            .unwrap()
            .to_string();
        assert!(refused.contains("is paired"), "{refused}");
    }

    #[test]
    fn accept_first_pairs_the_first_controller_only() {
        let harness = harness(2);
        pair_accept_first(&harness);
        assert!(!harness.agent.dir.accept_first());
        let other = Key::from_seed(&[8; 32]).unwrap();
        let mut late = connect(&harness, &other);
        assert!(
            late.request(frame::PAIR, &json!({"controllers": record(&other)}))
                .is_err()
        );
    }

    #[test]
    fn controllers_advance_through_signed_records_and_ranks_run_under_grants() {
        let harness = harness(3);
        let mut session = pair_accept_first(&harness);
        let second = Key::from_seed(&[4; 32]).unwrap();
        let revision2 = Signed::sign(
            &harness.controller,
            &ControllerSet {
                revision: 2,
                keys: vec![ControllerKey {
                    name: "second".into(),
                    public_key: second.public().to_hex(),
                }],
            },
        )
        .unwrap();
        session
            .write_frame(frame::CONTROLLERS, &serde_json::to_vec(&revision2).unwrap())
            .unwrap();
        session.expect(frame::OK).unwrap();
        assert!(session.request(frame::STATUS, &json!({})).is_err());
        let removed = Harness {
            address: harness.address.clone(),
            agent: harness.agent.clone(),
            controller: Key::from_seed(&[3; 32]).unwrap(),
            temp: tempfile::tempdir().unwrap(),
        };
        let harness = Harness {
            controller: second,
            ..harness
        };
        let mut session = connect(&harness, &harness.controller);
        session.request(frame::STATUS, &json!({})).unwrap();

        let script = "#!/bin/sh\necho \"$@\"\necho err >&2\nexit 3\n";
        let name = cache::binary_name(script.as_bytes());
        put(&mut session, "binary", &name, script.as_bytes(), None);
        assert_eq!(
            session
                .request(
                    frame::HAVE,
                    &json!({"kind": "binary", "name": name, "size": script.len()})
                )
                .unwrap(),
            [1]
        );
        put(
            &mut session,
            "package",
            "export-13/model.safetensors",
            b"package bytes",
            None,
        );
        let grant = grant_for(&harness, 0, None);
        let wrong_rank = session
            .request(frame::RUN, &run_request(&grant, 1, &name))
            .err()
            .unwrap()
            .to_string();
        assert!(wrong_rank.contains("no rank 1"), "{wrong_rank}");
        let mut session = connect(&harness, &harness.controller);
        let stale = grant_for(&removed, 0, None);
        assert!(
            session
                .request(frame::RUN, &run_request(&stale, 0, &name))
                .is_err()
        );
        let mut session = connect(&harness, &harness.controller);
        let reply: Value = serde_json::from_slice(
            &session
                .request(frame::RUN, &run_request(&grant, 0, &name))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(reply["group"], "test-group");
        let (lines, code) = collect(&mut session);
        assert_eq!(code, 3);
        let command = lines.iter().find(|l| l.starts_with("run ")).unwrap();
        assert!(
            command.contains(
                "export-13/model.safetensors --rank 0 --rendezvous 0.0.0.0:29800 \
                 --max-concurrency 2 --port 8001 --publisher-keys"
            ),
            "{command}"
        );
        assert!(command.ends_with(&format!(
            "--dangerously-approve-key {} --dangerously-accept-publisher test-publisher",
            publisher().fingerprint()
        )));
        assert!(lines.contains(&"err".to_owned()));

        let mut session = connect(&harness, &harness.controller);
        session
            .request(frame::ATTACH, &json!({"group": "test-group"}))
            .unwrap();
        assert_eq!(collect(&mut session), (lines, 3));
        let mut session = connect(&harness, &harness.controller);
        let status: Value =
            serde_json::from_slice(&session.request(frame::STATUS, &json!({})).unwrap()).unwrap();
        assert_eq!(status["group"]["running"], false);
        assert_eq!(status["group"]["exit"], 3);
        let stopped: Value = serde_json::from_slice(
            &session
                .request(frame::STOP, &json!({"group": "test-group"}))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(stopped["stopped"], false);
    }

    #[test]
    fn running_groups_are_stopped_and_outlive_their_controller_session() {
        let harness = harness(5);
        let mut session = pair_accept_first(&harness);
        let script = "#!/bin/sh\necho started\ntrap 'echo stopped; exit 7' TERM\nwhile true; do sleep 0.1; done\n";
        let name = cache::binary_name(script.as_bytes());
        put(&mut session, "binary", &name, script.as_bytes(), None);
        put(
            &mut session,
            "package",
            "export-13-rank1/model.safetensors",
            b"package bytes",
            Some(&[(0, 5), (8, 5)]),
        );
        let grant = grant_for(&harness, 1, None);
        session
            .request(frame::RUN, &run_request(&grant, 1, &name))
            .unwrap();
        let first = loop {
            let (tag, body) = session.read_frame().unwrap();
            if tag == frame::LOG && !body.is_empty() {
                break String::from_utf8(body).unwrap();
            }
        };
        assert_eq!(first, "started");
        drop(session);
        thread::sleep(Duration::from_millis(300));
        let mut session = connect(&harness, &harness.controller);
        let status: Value =
            serde_json::from_slice(&session.request(frame::STATUS, &json!({})).unwrap()).unwrap();
        assert_eq!(status["group"]["running"], true);
        let busy = session
            .request(
                frame::RUN,
                &run_request(&grant_for(&harness, 1, None), 1, &name),
            )
            .err()
            .unwrap()
            .to_string();
        assert!(busy.contains("is running"), "{busy}");
        let mut attached = connect(&harness, &harness.controller);
        attached.request(frame::ATTACH, &json!({})).unwrap();
        let mut session = connect(&harness, &harness.controller);
        let stopped: Value =
            serde_json::from_slice(&session.request(frame::STOP, &json!({})).unwrap()).unwrap();
        assert_eq!(stopped["stopped"], true);
        let (lines, code) = collect(&mut attached);
        assert_eq!(code, 7);
        assert!(
            lines.contains(&"started".to_owned()) && lines.contains(&"stopped".to_owned()),
            "{lines:?}"
        );
        // The group's pins end with it, so a later package replaces it.
        put(
            &mut session,
            "package",
            "other-1/model.safetensors",
            b"x",
            None,
        );
        assert!(
            !harness
                .agent
                .cache
                .join("packages/export-13-rank1")
                .exists()
        );
    }

    #[test]
    fn nodes_fetch_package_ranges_from_a_source_node_under_a_grant() {
        let source = harness(6);
        let peer = Harness {
            controller: Key::from_seed(&[6; 32]).unwrap(),
            ..harness(7)
        };
        pair_accept_first(&source);
        pair_accept_first(&peer);
        let package = source.temp.path().join("held/model.safetensors");
        fs::create_dir_all(package.parent().unwrap()).unwrap();
        let bytes: Vec<u8> = (0..100_000u32).map(|i| (i % 253) as u8).collect();
        fs::write(&package, &bytes).unwrap();
        let grant = Signed::sign(
            &source.controller,
            &Grant {
                group: "g".into(),
                package: Package {
                    export: "export".into(),
                    length: bytes.len() as u64,
                    files: vec!["model.safetensors".into()],
                    source: Some(Source {
                        public_key: source.agent.key.public().to_hex(),
                        path: package.display().to_string(),
                    }),
                },
                members: vec![
                    Member {
                        rank: 0,
                        name: "source".into(),
                        public_key: source.agent.key.public().to_hex(),
                        address: source.address.clone(),
                    },
                    Member {
                        rank: 1,
                        name: "peer".into(),
                        public_key: peer.agent.key.public().to_hex(),
                        address: peer.address.clone(),
                    },
                ],
                rendezvous: "127.0.0.1:29800".into(),
                serving: vec![],
                rank_zero: vec![],
                roce_ring: false,
                issued: now(),
                expires: now() + 60,
            },
        )
        .unwrap();
        let ranges = [(0u64, 4096u64), (50_000, 10_000)];
        let mut session = connect(&peer, &peer.controller);
        let reply: Value = serde_json::from_slice(
            &session
                .request(
                    frame::TRANSFER,
                    &json!({
                        "grant": grant,
                        "source": {"public_key": source.agent.key.public().to_hex(), "address": source.address},
                        "directory": "export-100000-rank1",
                        "files": [{"name": "model.safetensors", "size": bytes.len(), "ranges": ranges}],
                    }),
                )
                .unwrap(),
        )
        .unwrap();
        assert_eq!(reply["received"], 14096);
        let copy = fs::read(
            peer.agent
                .cache
                .join("packages/export-100000-rank1/model.safetensors"),
        )
        .unwrap();
        assert_eq!(copy.len(), bytes.len());
        for &(offset, nbytes) in &ranges {
            let span = offset as usize..(offset + nbytes) as usize;
            assert_eq!(copy[span.clone()], bytes[span]);
        }
        assert!(copy[4096..50_000].iter().all(|&b| b == 0));

        let stranger = Key::from_seed(&[9; 32]).unwrap();
        let mut session = Session::connect(&source.address, &stranger).unwrap();
        exchange_hello(&mut session, &controller_hello(&stranger)).unwrap();
        session
            .write_frame(
                frame::FETCH,
                &json!({"grant": grant, "name": "model.safetensors", "ranges": [[0, 10]]})
                    .to_string()
                    .into_bytes(),
            )
            .unwrap();
        assert!(
            session
                .expect(frame::OK)
                .err()
                .unwrap()
                .to_string()
                .contains("not a member")
        );
        let mut session = Session::connect(&source.address, &peer.agent.key).unwrap();
        exchange_hello(&mut session, &peer.agent.hello().unwrap()).unwrap();
        session
            .write_frame(
                frame::FETCH,
                &json!({"grant": grant, "name": "other", "ranges": [[0, 10]]})
                    .to_string()
                    .into_bytes(),
            )
            .unwrap();
        assert!(session.expect(frame::OK).is_err());

        let mut session = connect(&source, &source.controller);
        session
            .write_frame(
                frame::GET,
                &json!({"path": package, "offset": 50_000, "nbytes": 16})
                    .to_string()
                    .into_bytes(),
            )
            .unwrap();
        let (tag, length) = session.read_header().unwrap();
        assert_eq!((tag, length), (frame::OK, 16));
        assert_eq!(session.read_body(16, 16).unwrap(), bytes[50_000..50_016]);
        assert!(
            session
                .request(frame::STAT, &json!({"path": "relative"}))
                .is_err()
        );
    }
}
