//! Start a multi-host package's peer ranks from rank zero.
//!
//! Each peer host receives the package and this executable in its flare cache (rsync skips what
//! it already holds), then runs its rank over ssh. The executable is self-contained apart from
//! glibc, libibverbs and the CUDA driver, which every host provides. Peer output is forwarded
//! with a rank prefix; a peer exits once rank zero's rendezvous connection closes.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{BufRead, BufReader, Read};
use std::net::{IpAddr, ToSocketAddrs, UdpSocket};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::thread;

type Failure = Box<dyn std::error::Error + Send + Sync>;

/// Peer host directory holding replicated packages and executables, relative to its home.
const CACHE: &str = ".cache/flare";

/// The started peer ranks; dropping this stops any that are still running.
pub struct Peers {
    children: Vec<Child>,
}

impl Drop for Peers {
    fn drop(&mut self) {
        for child in &mut self.children {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// The address peers reach this host at: the local end of the route toward `host`.
pub fn address_toward(host: &str) -> Result<IpAddr, Failure> {
    let target = (host, 22)
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| format!("cannot resolve {host}"))?;
    let socket = UdpSocket::bind(if target.is_ipv4() {
        "0.0.0.0:0"
    } else {
        "[::]:0"
    })?;
    socket.connect(target)?;
    Ok(socket.local_addr()?.ip())
}

fn ssh(host: &str) -> Command {
    let mut command = Command::new("ssh");
    command.args(["-o", "BatchMode=yes", "-o", "ServerAliveInterval=15", host]);
    command
}

fn run(command: &mut Command, what: &str) -> Result<(), Failure> {
    let status = command.status().map_err(|e| format!("{what}: {e}"))?;
    if !status.success() {
        return Err(format!("{what} failed ({status})").into());
    }
    Ok(())
}

fn quote(argument: &str) -> String {
    format!("'{}'", argument.replace('\'', r"'\''"))
}

/// Copy the package and this executable into `host`'s flare cache unless it already holds them.
fn replicate(
    host: &str,
    package: &Path,
    export_id: &str,
    binary: &Path,
    binary_name: &str,
) -> Result<(String, String), Failure> {
    run(
        ssh(host).arg(format!("mkdir -p {CACHE}/packages {CACHE}/bin")),
        &format!("{host}: create the flare cache"),
    )?;
    let remote_binary = format!("{CACHE}/bin/{binary_name}");
    let remote_package = format!("{CACHE}/packages/{export_id}.safetensors");
    for (source, destination, what) in [
        (binary, &remote_binary, "copy flare"),
        // rsync's size and modification-time check skips a package the host already holds.
        (package, &remote_package, "copy the package"),
    ] {
        run(
            Command::new("rsync")
                .args(["-a", "--partial", "-e", "ssh -o BatchMode=yes"])
                .arg(source)
                .arg(format!("{host}:{destination}")),
            &format!("{host}: {what}"),
        )?;
    }
    Ok((remote_binary, remote_package))
}

fn forward(stream: impl Read + Send + 'static, prefix: String) {
    thread::spawn(move || {
        for line in BufReader::new(stream).lines().map_while(Result::ok) {
            eprintln!("[{prefix}] {line}");
        }
    });
}

/// Replicate the package and this executable to `hosts` (ranks 1 onward, in order) and start
/// their ranks, which join rank zero at `rendezvous` with the same serving `options`.
pub fn start(
    hosts: &[String],
    package: &Path,
    export_id: &str,
    rendezvous: &str,
    options: &[String],
) -> Result<Peers, Failure> {
    let binary = std::env::current_exe()?;
    let mut hasher = DefaultHasher::new();
    std::fs::read(&binary)?.hash(&mut hasher);
    let binary_name = format!("flare-{:016x}", hasher.finish());
    let copies: Vec<_> = hosts
        .iter()
        .map(|host| {
            let (host, package, export_id) =
                (host.clone(), package.to_owned(), export_id.to_owned());
            let (binary, binary_name) = (binary.clone(), binary_name.clone());
            thread::spawn(move || replicate(&host, &package, &export_id, &binary, &binary_name))
        })
        .collect();
    let mut targets = Vec::new();
    for (host, copy) in hosts.iter().zip(copies) {
        targets.push(
            copy.join()
                .map_err(|_| format!("{host}: copy thread failed"))??,
        );
        eprintln!("{host}: package and flare are in place");
    }
    let mut peers = Peers {
        children: Vec::new(),
    };
    for (index, (host, (remote_binary, remote_package))) in hosts.iter().zip(targets).enumerate() {
        let rank = index + 1;
        let mut arguments = vec![
            remote_binary,
            "run".into(),
            remote_package,
            "--rank".into(),
            rank.to_string(),
            "--rendezvous".into(),
            rendezvous.into(),
        ];
        arguments.extend(options.iter().cloned());
        let command = arguments
            .iter()
            .map(|a| quote(a))
            .collect::<Vec<_>>()
            .join(" ");
        let mut child = ssh(host)
            .arg(command)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("{host}: start rank {rank}: {e}"))?;
        forward(child.stdout.take().unwrap(), format!("rank {rank} {host}"));
        forward(child.stderr.take().unwrap(), format!("rank {rank} {host}"));
        peers.children.push(child);
    }
    Ok(peers)
}
