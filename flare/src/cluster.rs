//! Ranks of one model on separate hosts: rendezvous at rank zero and connect their RoCE transport.

use std::sync::Arc;
use std::time::Duration;

use flare_helios::{
    execution::{CollectiveHealth, RankStream},
    runtime_package::{ClusterRank, NativeModel},
    startup::StartupProgress,
};

type Failure = Box<dyn std::error::Error>;

// Ranks join before loading; rank zero starts its peers first, so the slowest start bounds the wait.
const JOIN_TIMEOUT: Duration = Duration::from_secs(1800);

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
pub use flare_helios::execution::roce::RoceTransport;

#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
pub struct RoceTransport;

/// A rank's connection to the rest of its cluster; the transport must outlive every replay.
pub struct Joined<S> {
    pub streams: S,
    pub transport: Arc<RoceTransport>,
}

/// The check every completed frame passes: a failed transport poisons the rank.
pub fn health(transport: &Arc<RoceTransport>) -> CollectiveHealth {
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        let transport = transport.clone();
        CollectiveHealth(Arc::new(move || transport.health()))
    }
    #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
    {
        let _ = transport;
        unreachable!("no transport without RoCE support")
    }
}

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
fn open(owner: &NativeModel) -> Result<RoceTransport, Failure> {
    let spec = owner
        .roce(0)
        .ok_or("multi-host package rank has no RoCE transport")?;
    // SAFETY: The package restored the region at its recorded address; owner outlives the
    // transport, and no graph runs before every rank connected.
    Ok(unsafe { RoceTransport::open(spec)? })
}

/// A rank's admitted connections to the rest of its cluster, made before it loads.
pub enum Membership {
    /// Rank zero's connection to each peer, ranks 1 onward.
    Zero(Vec<RankStream>),
    /// A peer's connection to rank zero.
    Peer(RankStream),
}

/// Join rank zero's rendezvous at `address` before loading: rank zero admits every peer of its
/// export, and a peer knocks until admitted.
pub fn join(
    rank: usize,
    world: usize,
    export_id: &str,
    address: &str,
    progress: &StartupProgress,
) -> Result<Membership, Failure> {
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        use flare_helios::execution::rendezvous;

        if rank == 0 {
            let phase = progress.phase("Wait for tensor-parallel hosts", None);
            let streams = rendezvous::accept(address, world, export_id, JOIN_TIMEOUT, |rank| {
                eprintln!("Rank {rank} joined")
            })?;
            phase.finish(format!("{} peer ranks at {address}", streams.len()));
            Ok(Membership::Zero(streams))
        } else {
            let phase = progress.phase("Join rank zero", None);
            let stream = rendezvous::knock(address, rank, world, export_id, JOIN_TIMEOUT)?;
            phase.finish(format!("rank {rank} of {world} at {address}"));
            Ok(Membership::Peer(stream))
        }
    }
    #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
    {
        let _ = (rank, world, export_id, address, progress);
        Err("this flare build has no RoCE transport for multi-host packages".into())
    }
}

impl Membership {
    /// The continuation rows and KV pages every rank allocates: the fewest any rank can back.
    pub fn agree(&mut self, local: (u64, u64)) -> flare_helios::Result<(u64, u64)> {
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        {
            use flare_helios::execution::rendezvous;

            match self {
                Membership::Zero(peers) => rendezvous::agree_capacity(peers, local),
                Membership::Peer(stream) => rendezvous::propose_capacity(stream, local),
            }
        }
        #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
        {
            let _ = local;
            unreachable!("no membership without RoCE support")
        }
    }
}

/// Rank zero: exchange transport records with every admitted peer and wait until every rank's
/// transport is connected.
pub fn rank_zero(
    owner: &NativeModel,
    mut streams: Vec<RankStream>,
    progress: &StartupProgress,
) -> Result<Joined<Vec<RankStream>>, Failure> {
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        use flare_helios::execution::rendezvous;

        let mut transport = open(owner)?;
        let phase = progress.phase("Connect RoCE transport", None);
        let records = rendezvous::share_records(&mut streams, transport.local_record()?)?;
        transport.connect(&records)?;
        rendezvous::await_connected(&mut streams)?;
        phase.finish(format!(
            "{} rails per peer over {}",
            transport.spec().rails,
            transport.spec().hcas.join(",")
        ));
        Ok(Joined {
            streams,
            transport: Arc::new(transport),
        })
    }
    #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
    {
        let _ = (owner, streams, progress);
        Err("this flare build has no RoCE transport for multi-host packages".into())
    }
}

/// A peer rank: exchange transport records with rank zero over its admitted stream, and connect.
pub fn peer(
    owner: &NativeModel,
    cluster: &ClusterRank,
    mut stream: RankStream,
    progress: &StartupProgress,
) -> Result<Joined<RankStream>, Failure> {
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        use flare_helios::execution::rendezvous;

        let mut transport = open(owner)?;
        let phase = progress.phase("Connect RoCE transport", None);
        let records =
            rendezvous::exchange_record(&mut stream, &transport.local_record()?, cluster.world)?;
        transport.connect(&records)?;
        rendezvous::report_connected(&mut stream)?;
        phase.finish(format!(
            "{} rails per peer over {}",
            transport.spec().rails,
            transport.spec().hcas.join(",")
        ));
        Ok(Joined {
            streams: stream,
            transport: Arc::new(transport),
        })
    }
    #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
    {
        let _ = (owner, cluster, stream, progress);
        Err("this flare build has no RoCE transport for multi-host packages".into())
    }
}
