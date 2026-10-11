//! A model package owns a disjoint resource bundle for every tensor-parallel rank.

use super::{
    NativeResources,
    driver::{Access, Location},
    format::{Memory, Package, Reservation, invalid},
    protobuf,
};
use crate::{Result, execution::ExecutionManifest, startup::StartupProgress};
use prost::Message;
use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    thread,
};

pub struct NativeModel {
    ranks: Vec<NativeResources>,
    frontend_root: PathBuf,
}

impl NativeModel {
    /// # Safety
    /// All rank bundles must be trusted and obey the CUDA execution contract.
    pub unsafe fn load(root: &Path, devices: &[i32], progress: &StartupProgress) -> Result<Self> {
        let phase = progress.phase("Prepare model", None);
        let descriptor = protobuf::wire::ModelPackage::decode(
            fs::read(root.join("model.pb"))
                .map_err(|e| e.to_string())?
                .as_slice(),
        )
        .map_err(|e| invalid(format!("invalid model package: {e}")))?;
        if descriptor.version != 1
            || ![1, 2, 4].contains(&descriptor.ranks.len())
            || devices.len() != descriptor.ranks.len()
            || devices.iter().collect::<HashSet<_>>().len() != devices.len()
        {
            return Err(invalid(
                "model rank count must match distinct selected devices",
            ));
        }
        let mut paths = Vec::new();
        let mut packages: Vec<Package> = Vec::new();
        for (rank, path) in descriptor.ranks.iter().enumerate() {
            if path != &format!("rank-{rank}") {
                return Err(invalid("rank bundle paths must be rank-N in rank order"));
            }
            let path = root.join(path);
            let package =
                protobuf::decode(&fs::read(path.join("package.pb")).map_err(|e| e.to_string())?)?;
            package.validate()?;
            paths.push(path);
            packages.push(package);
        }
        let mut allocations = Vec::new();
        let mut ranges = Vec::new();
        for package in &mut packages {
            // Peer allocations name storage another rank owns and are checked against model.pb below.
            for allocation in package
                .allocations
                .iter()
                .filter(|a| !matches!(a.memory, Memory::Peer))
            {
                let address = allocation
                    .address
                    .ok_or_else(|| invalid("model allocations require recorded addresses"))?;
                let end = address
                    .checked_add(allocation.nbytes as u64)
                    .ok_or_else(|| invalid("allocation range overflow"))?;
                allocations.push((address, end));
            }
            for reservation in package.reservations.drain(..) {
                ranges.push((
                    reservation.address,
                    reservation.address + reservation.nbytes as u64,
                ));
            }
        }
        allocations.sort_unstable();
        if allocations.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(invalid("rank allocation addresses overlap"));
        }
        ranges.sort_unstable();
        let mut merged: Vec<Reservation> = Vec::new();
        for (start, end) in ranges {
            if let Some(previous) = merged
                .last_mut()
                .filter(|r| start <= r.address + r.nbytes as u64)
            {
                previous.nbytes = (end.max(previous.address + previous.nbytes as u64)
                    - previous.address) as usize;
            } else {
                merged.push(Reservation {
                    address: start,
                    nbytes: (end - start) as usize,
                });
            }
        }
        packages[0].reservations = merged;
        let mut shared = HashSet::new();
        for peer in &descriptor.peer_memory {
            let owner = packages
                .get(peer.owner as usize)
                .ok_or_else(|| invalid("invalid peer-memory owner"))?;
            if !shared.insert(peer.address)
                || !peer.readers.contains(&peer.owner)
                || peer.readers.iter().collect::<HashSet<_>>().len() != peer.readers.len()
                || peer.readers.is_empty()
                || peer.readers.iter().any(|r| *r as usize >= devices.len())
                || !owner.allocations.iter().any(|a| {
                    a.address == Some(peer.address)
                        && a.nbytes as u64 == peer.nbytes
                        && matches!(a.memory, Memory::Device)
                })
            {
                return Err(invalid(
                    "peer memory must name one owned device allocation and valid readers",
                ));
            }
        }
        for (rank, package) in packages.iter().enumerate() {
            for allocation in package
                .allocations
                .iter()
                .filter(|a| matches!(a.memory, Memory::Peer))
            {
                if !descriptor.peer_memory.iter().any(|peer| {
                    peer.owner as usize != rank
                        && Some(peer.address) == allocation.address
                        && peer.nbytes == allocation.nbytes as u64
                        && peer.readers.contains(&(rank as u32))
                }) {
                    return Err(invalid(
                        "peer allocation must name another rank's shared memory this rank reads",
                    ));
                }
            }
        }
        phase.finish(format!(
            "{} rank{}",
            devices.len(),
            if devices.len() == 1 { "" } else { "s" }
        ));
        let mut model = Self {
            ranks: Vec::new(),
            frontend_root: paths[0].clone(),
        };
        unsafe {
            let phase = progress.phase(
                if devices.len() == 1 {
                    "Initialize GPU"
                } else {
                    "Initialize GPUs"
                },
                None,
            );
            for (package, device) in packages.iter().zip(devices) {
                model
                    .ranks
                    .push(NativeResources::initialize(package, *device)?);
            }
            phase.finish(
                packages
                    .iter()
                    .zip(devices)
                    .map(|(package, device)| {
                        format!(
                            "GPU {device} · SM{}{}",
                            package.architecture[0], package.architecture[1]
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" / "),
            );
            let (mut gpu_bytes, mut host_bytes, mut payload_bytes) = (0u64, 0u64, 0u64);
            for allocation in packages.iter().flat_map(|package| &package.allocations) {
                match allocation.memory {
                    Memory::Device => gpu_bytes += allocation.nbytes as u64,
                    Memory::MappedHost => host_bytes += allocation.nbytes as u64,
                    Memory::Peer => {}
                }
                payload_bytes += allocation.stored_bytes() as u64;
            }
            let phase = progress.phase("Allocate model memory", Some(gpu_bytes + host_bytes));
            // Reservations cover all ranks and must exist before parallel mapping.
            model.ranks[0].reserve(&packages[0])?;
            model.parallel(&packages, &paths, |rank, package, _| {
                rank.allocate(package, &phase)
            })?;
            phase.finish(format!(
                "{:.1} GiB GPU · {:.1} GiB host",
                gpu_bytes as f64 / (1u64 << 30) as f64,
                host_bytes as f64 / (1u64 << 30) as f64,
            ));
            let phase = progress.phase("Load weights", Some(payload_bytes));
            phase.advance_bytes(0);
            model.parallel(&packages, &paths, |rank, package, path| {
                rank.load_payload(path, package, &phase, devices.len())
            })?;
            phase.finish(format!(
                "{:.1} GiB",
                payload_bytes as f64 / (1u64 << 30) as f64
            ));
            for peer in &descriptor.peer_memory {
                let rank = &model.ranks[peer.owner as usize];
                let access: Vec<_> = peer
                    .readers
                    .iter()
                    .map(|reader| Access {
                        location: Location {
                            kind: 1,
                            device: model.ranks[*reader as usize].device,
                        },
                        flags: 3,
                    })
                    .collect();
                rank.driver
                    .check("cuCtxSetCurrent", (rank.driver.context_set)(rank.context))?;
                rank.driver.check(
                    "cuMemSetAccess",
                    (rank.driver.access)(
                        peer.address,
                        peer.nbytes as usize,
                        access.as_ptr(),
                        access.len(),
                    ),
                )?;
            }
            let kernels = packages
                .iter()
                .map(|p| p.functions.len() as u64)
                .sum::<u64>();
            let modules = packages.iter().map(|p| p.modules.len() as u64).sum::<u64>();
            let phase = progress.phase("Prepare kernels", Some(kernels + modules));
            model.parallel(&packages, &paths, |rank, package, path| {
                rank.prepare_kernels(path, package, &phase)
            })?;
            phase.finish(format!("{kernels} kernels"));
            let nodes = packages
                .iter()
                .flat_map(|p| &p.graphs)
                .map(|graph| graph.nodes.len() as u64 + 1)
                .sum();
            let phase = progress.phase("Prepare execution", Some(nodes));
            model.parallel(&packages, &paths, |rank, package, _| {
                rank.prepare_graphs(package, &phase)
            })?;
            phase.finish(
                packages[0]
                    .execution
                    .as_ref()
                    .map(|execution| format!("concurrency {}", execution.max_active))
                    .unwrap_or_default(),
            );
        }
        Ok(model)
    }

    fn parallel(
        &mut self,
        packages: &[Package],
        paths: &[PathBuf],
        operation: impl Fn(&mut NativeResources, &Package, &Path) -> Result<()> + Sync,
    ) -> Result<()> {
        thread::scope(|scope| {
            let mut workers = Vec::new();
            for (index, ((rank, package), path)) in
                self.ranks.iter_mut().zip(packages).zip(paths).enumerate()
            {
                let operation = &operation;
                workers.push(
                    thread::Builder::new()
                        .name(format!("helios-load-{index}"))
                        .spawn_scoped(scope, move || {
                            operation(rank, package, path)
                                .map_err(|error| invalid(format!("rank {index}: {error}")))
                        })
                        .map_err(|error| error.to_string())?,
                );
            }
            // Join every rank before any error can release shared reservations.
            let results: Vec<_> = workers
                .into_iter()
                .map(|worker| {
                    worker
                        .join()
                        .unwrap_or_else(|_| Err(invalid("rank loader panicked")))
                })
                .collect();
            results.into_iter().collect()
        })
    }

    pub fn manifests(&self) -> Result<Vec<ExecutionManifest>> {
        self.ranks
            .iter()
            .map(NativeResources::execution_manifest)
            .collect()
    }

    pub fn frontend_root(&self) -> &Path {
        &self.frontend_root
    }
}

impl Drop for NativeModel {
    fn drop(&mut self) {
        // Rank zero owns the shared reservations and releases them after every peer.
        while self.ranks.pop().is_some() {}
    }
}
