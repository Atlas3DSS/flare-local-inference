//! Ranks of one model on separate hosts: rendezvous at rank zero and connect their RoCE transport.

use std::sync::Arc;
use std::time::Duration;

use flare_helios::{
    execution::{CollectiveHealth, RankStream},
    runtime_package::{ClusterRank, NativeModel},
    startup::StartupProgress,
};

type Failure = Box<dyn std::error::Error>;

// Every host loads its rank before joining; the slowest load bounds the wait.
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

/// Rank zero: admit every peer at `address`, exchange transport records, and wait until every
/// rank's transport is connected.
pub fn rank_zero(
    owner: &NativeModel,
    cluster: &ClusterRank,
    address: &str,
    progress: &StartupProgress,
) -> Result<Joined<Vec<RankStream>>, Failure> {
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        use flare_helios::execution::rendezvous;

        let mut transport = open(owner)?;
        let phase = progress.phase("Wait for tensor-parallel hosts", None);
        let mut streams = rendezvous::accept(
            address,
            cluster.world,
            &cluster.export_id,
            JOIN_TIMEOUT,
            |rank| eprintln!("Rank {rank} joined"),
        )?;
        phase.finish(format!("{} peer ranks at {address}", streams.len()));
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
        let _ = (owner, cluster, address, progress);
        Err("this flare build has no RoCE transport for multi-host packages".into())
    }
}

/// A peer rank: join rank zero at `address`, exchange transport records, and connect.
pub fn peer(
    owner: &NativeModel,
    cluster: &ClusterRank,
    address: &str,
    progress: &StartupProgress,
) -> Result<Joined<RankStream>, Failure> {
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        use flare_helios::execution::rendezvous;

        let mut transport = open(owner)?;
        let phase = progress.phase("Join rank zero", None);
        let mut stream = rendezvous::knock(
            address,
            cluster.rank,
            cluster.world,
            &cluster.export_id,
            JOIN_TIMEOUT,
        )?;
        phase.finish(format!(
            "rank {} of {} at {address}",
            cluster.rank, cluster.world
        ));
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
        let _ = (owner, cluster, address, progress);
        Err("this flare build has no RoCE transport for multi-host packages".into())
    }
}
