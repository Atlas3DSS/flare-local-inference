//! Preparation checkpoint channels through authenticated agents on the physical ring.

use std::io::{Read, Write};
use std::net::{IpAddr, Ipv4Addr, Shutdown, SocketAddr, TcpStream};
use std::os::fd::FromRawFd;
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::agent::{Hello, exchange_hello};
use super::fabric::{self, Link};
use super::grant::{Member, Signed};
use super::identity::{Key, PublicKey};
use super::session::{RECORD_PLAINTEXT, Session};
use super::store::AgentDir;
use super::{Failure, frame, now};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteRequest {
    pub gpu_uuids: Vec<String>,
    pub source_port: u16,
    pub streams: usize,
    pub catalog_sha256: String,
    pub identity: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Grant {
    pub purpose: String,
    pub members: Vec<Member>,
    pub ring: Vec<Vec<Link>>,
    pub source_port: u16,
    pub streams: usize,
    pub catalog_sha256: String,
    pub identity: String,
    pub expires: u64,
}

impl Grant {
    pub fn validate(&self) -> Result<(), Failure> {
        if self.purpose != "checkpoint-stream-v1"
            || self.source_port == 0
            || !(1..=64).contains(&self.streams)
            || now() > self.expires
        {
            return Err("invalid or expired checkpoint transport grant".into());
        }
        super::unhex32(&self.catalog_sha256)?;
        super::unhex32(&self.identity)?;
        fabric::validate_ring(&self.members, &self.ring)
    }

    fn rank(&self, key: PublicKey) -> Result<usize, Failure> {
        self.members
            .iter()
            .position(|m| m.public_key == key.to_hex())
            .ok_or_else(|| "checkpoint node is outside the grant".into())
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Open {
    pub grant: Signed,
    pub destination: usize,
    pub stream: usize,
}

pub fn authorize(dir: &AgentDir, signed: &Signed) -> Result<Grant, Failure> {
    let (_, controllers) = dir
        .controllers()?
        .ok_or("checkpoint transport requires paired agents")?;
    let grant: Grant = signed.verify(|key| controllers.contains(key))?;
    grant.validate()?;
    Ok(grant)
}

fn route(grant: &Grant, request: &Open) -> Result<Vec<usize>, Failure> {
    if request.stream >= grant.streams {
        return Err("checkpoint stream is outside its grant".into());
    }
    fabric::path(grant.members.len(), 0, request.destination)
}

fn upstream(
    key: &Key,
    hello: &Hello,
    grant: &Grant,
    request: &Open,
    rank: usize,
) -> Result<Session, Failure> {
    let route = route(grant, request)?;
    let index = route
        .iter()
        .position(|r| *r == rank)
        .ok_or("checkpoint node is outside its route")?;
    let previous = *route
        .get(
            index
                .checked_sub(1)
                .ok_or("checkpoint source has no upstream")?,
        )
        .ok_or("checkpoint route lacks its upstream")?;
    let (local, remote) = fabric::ring_hop(
        &grant.members,
        &grant.ring,
        rank,
        previous,
        request.stream % 2,
    )?;
    let mut session = Session::connect_from(local, remote, key)?;
    let peer = exchange_hello(&mut session, hello)?;
    if peer.kind != "node"
        || session.remote() != PublicKey::from_hex(&grant.members[previous].public_key)?
    {
        return Err("checkpoint upstream identity differs from the signed route".into());
    }
    session.write_frame(frame::CHECKPOINT_OPEN, &serde_json::to_vec(request)?)?;
    if !session.expect(frame::OK)?.is_empty() {
        return Err("checkpoint channel admission returned an unexpected body".into());
    }
    Ok(session)
}

fn copy(mut source: impl Read, mut target: impl Write) -> Result<(), Failure> {
    let mut buffer = vec![0; RECORD_PLAINTEXT];
    loop {
        let count = match source.read(&mut buffer) {
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        if count == 0 {
            return Ok(());
        }
        target.write_all(&buffer[..count])?;
    }
}

/// Both directions drain until one closes or fails; shutdown wakes the other blocked reader.
fn bridge(
    left_read: impl Read + Send,
    left_write: impl Write + Send,
    right_read: impl Read + Send,
    right_write: impl Write + Send,
    stop: impl FnOnce(),
) -> Result<(), Failure> {
    thread::scope(|scope| {
        let (sender, receiver) = mpsc::channel();
        let second = sender.clone();
        let a = scope.spawn(move || {
            let _ = sender.send(copy(left_read, right_write));
        });
        let b = scope.spawn(move || {
            let _ = second.send(copy(right_read, left_write));
        });
        let result = receiver
            .recv()
            .map_err(|_| "checkpoint relay worker stopped")?;
        stop();
        a.join().map_err(|_| "checkpoint relay worker panicked")?;
        b.join().map_err(|_| "checkpoint relay worker panicked")?;
        result
    })
}

fn admit(session: &Session, key: &Key, grant: &Grant, request: &Open) -> Result<usize, Failure> {
    let route = route(grant, request)?;
    let rank = grant.rank(key.public())?;
    let index = route
        .iter()
        .position(|r| *r == rank)
        .ok_or("checkpoint relay is outside the route")?;
    let next = *route
        .get(index + 1)
        .ok_or("checkpoint destination cannot relay itself")?;
    if session.remote() != PublicKey::from_hex(&grant.members[next].public_key)? {
        return Err("checkpoint channel did not arrive from its next ring member".into());
    }
    let (from, to) = fabric::ring_hop(&grant.members, &grant.ring, next, rank, request.stream % 2)?;
    let (local, remote) = session.addresses()?;
    if local != SocketAddr::V4(to) || remote.ip() != IpAddr::V4(from) {
        return Err("checkpoint channel did not use its signed fabric interfaces".into());
    }
    Ok(rank)
}

fn admission<T>(session: &mut Session, result: Result<T, Failure>) -> Result<T, Failure> {
    if let Err(error) = &result {
        session.error(&error.to_string())?;
    }
    result
}

pub fn relay(
    mut session: Session,
    key: &Key,
    hello: &Hello,
    grant: Grant,
    request: Open,
) -> Result<(), Failure> {
    let result = admit(&session, key, &grant, &request);
    let rank = admission(&mut session, result)?;
    if rank == 0 {
        let result = TcpStream::connect_timeout(
            &SocketAddr::from((Ipv4Addr::LOCALHOST, grant.source_port)),
            Duration::from_secs(10),
        )
        .map_err(Into::into);
        let mut source = admission(&mut session, result)?;
        source.set_nodelay(true)?;
        // The source receives the agent-verified rank and stream before its end-to-end handshake.
        source.write_all(&(request.destination as u32).to_le_bytes())?;
        source.write_all(&(request.stream as u32).to_le_bytes())?;
        session.ok(&[])?;
        let (read, send, network) = session.checkpoint_parts()?;
        bridge(read, send, &source, &source, || {
            let _ = network.shutdown(Shutdown::Both);
            let _ = source.shutdown(Shutdown::Both);
        })
    } else {
        let result = upstream(key, hello, &grant, &request, rank);
        let upstream = admission(&mut session, result)?;
        session.ok(&[])?;
        let (read, send, downstream) = session.checkpoint_parts()?;
        let (up_read, up_send, upstream) = upstream.checkpoint_parts()?;
        bridge(read, send, up_read, up_send, || {
            let _ = downstream.shutdown(Shutdown::Both);
            let _ = upstream.shutdown(Shutdown::Both);
        })
    }
}

/// Private Ignition helper: the inherited Unix socket stays local; all inter-host bytes use the ring.
pub fn connect(path: &Path, rank: usize, stream: usize, descriptor: i32) -> Result<(), Failure> {
    if descriptor < 3 {
        return Err("checkpoint helper requires an inherited socket".into());
    }
    let dir = AgentDir::new(AgentDir::default_root());
    if !dir.root.join("node.key").is_file() {
        return Err("checkpoint transport requires this host's installed agent identity".into());
    }
    let signed: Signed = serde_json::from_slice(&std::fs::read(path)?)?;
    let grant = authorize(&dir, &signed)?;
    let key = dir.key()?;
    if grant.rank(key.public())? != rank {
        return Err("checkpoint rank differs from its paired agent identity".into());
    }
    let hello = Hello {
        version: super::PROTOCOL_VERSION,
        kind: "node".into(),
        name: grant.members[rank].name.clone(),
        public_key: key.public().to_hex(),
        controllers_revision: 0,
        free_bytes: 0,
        executable: None,
    };
    let request = Open {
        grant: signed,
        destination: rank,
        stream,
    };
    let session = upstream(&key, &hello, &grant, &request, rank)?;
    // SAFETY: Ignition passes an owned socketpair descriptor via pass_fds; this process owns its copy.
    let local = unsafe { UnixStream::from_raw_fd(descriptor) };
    let (read, send, network) = session.checkpoint_parts()?;
    bridge(&local, &local, read, send, || {
        let _ = local.shutdown(Shutdown::Both);
        let _ = network.shutdown(Shutdown::Both);
    })
}

#[cfg(test)]
mod tests {
    use super::super::agent::tests::{Harness, connect as controller, harness, pair_accept_first};
    use super::super::store::Node;
    use super::*;
    use serde_json::json;
    use std::net::TcpListener;

    #[test]
    fn checkpoint_streams_cross_authenticated_relays_and_reject_wrong_hops() {
        let hosts = (120..124)
            .map(|seed| Harness {
                controller: Key::from_seed(&[120; 32]).unwrap(),
                ..harness(seed)
            })
            .collect::<Vec<_>>();
        for host in &hosts {
            pair_accept_first(host);
        }
        let nodes = hosts
            .iter()
            .enumerate()
            .map(|(rank, h)| Node {
                name: format!("rank{rank}"),
                address: h.address.clone(),
                public_key: h.agent.key.public().to_hex(),
                fingerprint: h.agent.key.public().fingerprint(),
            })
            .collect::<Vec<_>>();
        let records = hosts
            .iter()
            .map(|h| {
                controller(h, &h.controller)
                    .request(frame::FABRIC, &json!({}))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let source = TcpListener::bind("127.0.0.1:0").unwrap();
        let grant = Grant {
            purpose: "checkpoint-stream-v1".into(),
            members: nodes
                .iter()
                .enumerate()
                .map(|(rank, n)| Member {
                    rank,
                    name: n.name.clone(),
                    public_key: n.public_key.clone(),
                    // No checkpoint bytes may use these deliberately unreachable management addresses.
                    address: "127.0.0.1:1".into(),
                    gpus: vec![],
                })
                .collect(),
            ring: fabric::plan(&nodes, &records).unwrap(),
            source_port: source.local_addr().unwrap().port(),
            streams: 8,
            identity: "ab".repeat(32),
            catalog_sha256: "cd".repeat(32),
            expires: now() + 60,
        };
        let signed = Signed::sign(&hosts[0].controller, &grant).unwrap();
        let bytes = (0..2_000_019).map(|i| (i % 251) as u8).collect::<Vec<_>>();
        thread::scope(|scope| {
            let source_bytes = &bytes;
            scope.spawn(move || {
                thread::scope(|workers| {
                    for _ in 0..24 {
                        let (mut stream, _) = source.accept().unwrap();
                        workers.spawn(move || {
                            stream
                                .set_read_timeout(Some(Duration::from_secs(10)))
                                .unwrap();
                            let mut slot = [0; 8];
                            stream.read_exact(&mut slot).unwrap();
                            assert!(
                                (1..=3)
                                    .contains(&u32::from_le_bytes(slot[..4].try_into().unwrap()))
                            );
                            assert!(u32::from_le_bytes(slot[4..].try_into().unwrap()) < 8);
                            let mut request = [0; 7];
                            stream.read_exact(&mut request).unwrap();
                            assert_eq!(&request, b"ranges!");
                            stream.write_all(source_bytes).unwrap();
                            stream.read_exact(&mut request).unwrap();
                            assert_eq!(&request, b"finish!");
                            // Export reverses bulk traffic through the same authenticated hops.
                            let mut exported = vec![0; source_bytes.len()];
                            stream.read_exact(&mut exported).unwrap();
                            assert_eq!(&exported, source_bytes);
                            stream.write_all(b"drained").unwrap();
                        });
                    }
                });
            });
            for rank in 1..4 {
                for stream in 0..8 {
                    let (grant, signed, bytes, host) = (&grant, &signed, &bytes, &hosts[rank]);
                    scope.spawn(move || {
                        let request = Open {
                            grant: signed.clone(),
                            destination: rank,
                            stream,
                        };
                        let session = upstream(
                            &host.agent.key,
                            &host.agent.hello().unwrap(),
                            grant,
                            &request,
                            rank,
                        )
                        .unwrap();
                        let (mut read, mut write, socket) = session.checkpoint_parts().unwrap();
                        write.write_all(b"ranges!").unwrap();
                        let mut received = vec![0; bytes.len()];
                        read.read_exact(&mut received).unwrap();
                        assert_eq!(&received, bytes);
                        write.write_all(b"finish!").unwrap();
                        write.write_all(bytes).unwrap();
                        let mut reply = [0; 7];
                        read.read_exact(&mut reply).unwrap();
                        assert_eq!(&reply, b"drained");
                        socket.shutdown(Shutdown::Both).unwrap();
                    });
                }
            }
        });
        let closed_source = Open {
            grant: signed.clone(),
            destination: 1,
            stream: 0,
        };
        assert!(
            upstream(
                &hosts[1].agent.key,
                &hosts[1].agent.hello().unwrap(),
                &grant,
                &closed_source,
                1,
            )
            .is_err()
        );
        let request = Open {
            grant: signed.clone(),
            destination: 2,
            stream: 0,
        };
        // Rank 2 must reach root through rank 1, even when it possesses a valid grant.
        let error = controller(&hosts[0], &hosts[2].agent.key)
            .request(frame::CHECKPOINT_OPEN, &json!(request))
            .unwrap_err();
        assert!(error.to_string().contains("next ring member"));
        let mut invalid = grant.clone();
        invalid.ring[0][0].from.set_ip(Ipv4Addr::new(127, 0, 0, 2));
        let request = Open {
            grant: Signed::sign(&hosts[0].controller, &invalid).unwrap(),
            destination: 2,
            stream: 0,
        };
        let error = controller(&hosts[0], &hosts[1].agent.key)
            .request(frame::CHECKPOINT_OPEN, &json!(request))
            .unwrap_err();
        assert!(error.to_string().contains("fabric interfaces"));
        let request = Open {
            grant: signed,
            destination: 2,
            stream: 8,
        };
        assert!(route(&grant, &request).is_err());
        invalid = grant.clone();
        invalid.expires = now() - 1;
        assert!(invalid.validate().is_err());
        let mut invalid = request.grant;
        invalid.body.push(' ');
        assert!(authorize(&hosts[0].agent.dir, &invalid).is_err());
    }
}
