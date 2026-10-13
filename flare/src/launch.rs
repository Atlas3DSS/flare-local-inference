//! Start a multi-host package's peer ranks from rank zero through each host's `flare agent`.
//!
//! Each peer host's agent receives the package and this executable into its flare cache unless it
//! already holds them, then runs the peer's rank with rank zero's serving options and rendezvous
//! address. Peer output is forwarded with a rank prefix; closing the connections stops the ranks.

use std::net::{IpAddr, TcpStream, ToSocketAddrs, UdpSocket};
use std::path::Path;
use std::thread;

use crate::agent::{self, Agent, Failure};

/// The started peer ranks; dropping this closes their agents' connections, which stops them.
pub struct Peers {
    _connections: Vec<TcpStream>,
}

/// The address peers reach this host at: the local end of the route toward `host`.
pub fn address_toward(host: &str) -> Result<IpAddr, Failure> {
    let target = (host.split(':').next().unwrap_or(host), agent::PORT)
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

/// Send the package and this executable to `hosts` (ranks 1 onward, in order) where missing, then
/// start their ranks joining rank zero at `rendezvous` with the same serving `options`. With
/// `shares`, each peer's byte ranges in host order, a current agent keeps only its rank's bytes.
#[allow(clippy::too_many_arguments)]
pub fn start(
    hosts: &[String],
    package: &Path,
    export_id: &str,
    rendezvous: &str,
    options: &[String],
    token: &[u8],
    publisher: &flare_helios::runtime_package::Publisher,
    shares: Option<&[Vec<(u64, u64)>]>,
) -> Result<Peers, Failure> {
    let binary = std::env::current_exe()?;
    let binary_name = agent::binary_name(&std::fs::read(&binary)?);
    let length = std::fs::metadata(package)?.len();
    // Connect and copy to every host in parallel; each copy is skipped where the host has it.
    let ready: Vec<_> = hosts
        .iter()
        .enumerate()
        .map(|(index, host)| {
            let (host, binary, binary_name) = (host.clone(), binary.clone(), binary_name.clone());
            let (package, export_id, token) =
                (package.to_owned(), export_id.to_owned(), token.to_vec());
            let share = shares.map(|shares| shares[index].clone());
            thread::spawn(move || -> Result<(Agent, String), Failure> {
                let rank = index + 1;
                let mut agent = Agent::connect(&host, &token)?;
                if agent.ensure("binary", &binary_name, &binary, None)? {
                    eprintln!("{host}: sent flare {binary_name}");
                }
                // A current agent keeps only the bytes its rank reads, under a name for that rank.
                let (name, ranges) = match share.filter(|_| agent.holds_ranges()) {
                    Some(ranges) => (
                        format!("{export_id}-{length}-rank{rank}.safetensors"),
                        Some(ranges),
                    ),
                    None => (format!("{export_id}-{length}.safetensors"), None),
                };
                if agent.ensure("package", &name, &package, ranges.as_deref())? {
                    let sent = ranges.map_or(length, |r| r.iter().map(|&(_, n)| n).sum());
                    eprintln!(
                        "{host}: sent rank {rank} of the package, {:.1} GiB",
                        sent as f64 / (1u64 << 30) as f64
                    );
                }
                Ok((agent, name))
            })
        })
        .collect();
    let mut agents = Vec::new();
    for (host, thread) in hosts.iter().zip(ready) {
        agents.push(
            thread
                .join()
                .map_err(|_| format!("{host}: transfer thread failed"))??,
        );
    }
    let mut connections = Vec::new();
    for (index, (host, (agent, package_name))) in hosts.iter().zip(agents).enumerate() {
        let rank = index + 1;
        let mut arguments = vec![
            "--rank".into(),
            rank.to_string(),
            "--rendezvous".into(),
            rendezvous.into(),
        ];
        arguments.extend(options.iter().cloned());
        let prefix = format!("rank {rank} {host}");
        connections.push(agent.run(
            &binary_name,
            &package_name,
            &arguments,
            publisher,
            move |line| eprintln!("[{prefix}] {line}"),
        )?);
    }
    Ok(Peers {
        _connections: connections,
    })
}
