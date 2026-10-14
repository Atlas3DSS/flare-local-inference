//! Choose physical cache capacities within reserved virtual address ranges.

use super::{
    NativeModel, NativeResources,
    format::{Package, invalid},
};
use crate::{
    Result,
    offload::{CacheGeometry, Plane},
    startup::StartupProgress,
};

// Graphs are uploaded before sizing caches; leave room for driver bookkeeping
// and the execution streams and events created when the server starts.
const HEADROOM_BYTES: u64 = 64 << 20;
// A GPU that allocates from system memory (GB10) shares it with the operating system and with
// the server's host allocations made after sizing: frontend workers, tokenizers and request
// buffers took about 1.4 GiB on a DGX Spark. Caches leave room for both beyond the reserve.
const INTEGRATED_HOST_BYTES: u64 = 2 << 30;
const INTEGRATED_SYSTEM_SHARE: u64 = 20;

#[cfg(any(unix, test))]
fn shared_available(meminfo: &str, total: usize) -> Result<usize> {
    let available = meminfo.lines().find_map(|line| {
        let mut fields = line.split_whitespace();
        if fields.next()? != "MemAvailable:" {
            return None;
        }
        let kib = fields.next()?.parse::<usize>().ok()?;
        (fields.next()? == "kB").then_some(kib)?.checked_mul(1024)
    });
    available
        .map(|n| n.min(total))
        .ok_or_else(|| invalid("Linux MemAvailable is required to size integrated GPU memory"))
}

impl NativeResources {
    /// Memory the driver reports in total for the rank's device: on an integrated GPU, the
    /// system's memory.
    pub(super) unsafe fn total_memory(&self) -> Result<usize> {
        let (mut free, mut total) = (0, 0);
        unsafe {
            self.driver
                .check("cuCtxSetCurrent", (self.driver.context_set)(self.context))?;
            self.driver.check(
                "cuMemGetInfo_v2",
                (self.driver.memory_info)(&mut free, &mut total),
            )?;
        }
        Ok(total)
    }

    pub(super) unsafe fn available_memory(&self) -> Result<usize> {
        let (mut free, mut total, mut integrated) = (0, 0, 0);
        unsafe {
            self.driver
                .check("cuCtxSetCurrent", (self.driver.context_set)(self.context))?;
            self.driver.check(
                "cuMemGetInfo_v2",
                (self.driver.memory_info)(&mut free, &mut total),
            )?;
            self.driver.check(
                "cuDeviceGetAttribute(INTEGRATED)",
                (self.driver.device_attribute)(&mut integrated, 18, self.device),
            )?;
        }
        if integrated == 0 {
            Ok(free)
        } else {
            #[cfg(windows)]
            {
                // CUDA includes both the dedicated carveout and shared system memory.
                Ok(free.min(total))
            }
            #[cfg(unix)]
            {
                let meminfo =
                    std::fs::read_to_string("/proc/meminfo").map_err(|e| e.to_string())?;
                shared_available(&meminfo, total)
            }
        }
    }
}

fn headroom(config: &serde_json::Value) -> Result<u64> {
    let reserve = config
        .get("reserve_bytes")
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| invalid("memory reserve must be a nonnegative integer"))
        })
        .transpose()?
        .unwrap_or(0);
    reserve
        .checked_add(HEADROOM_BYTES)
        .ok_or_else(|| invalid("memory reserve overflows the device budget"))
}

/// Memory caches may use on one rank: free memory less the headroom, and on an integrated GPU
/// also the server's later host allocations and a twentieth of memory for the system.
fn budget(free: u64, total: u64, headroom: u64, integrated: bool) -> u64 {
    let host = if integrated {
        INTEGRATED_HOST_BYTES + total / INTEGRATED_SYSTEM_SHARE
    } else {
        0
    };
    free.saturating_sub(headroom.saturating_add(host))
}

fn bytes(package: &Package, capacity: i32, units: u64) -> Result<u64> {
    package
        .allocations
        .iter()
        .filter_map(|a| {
            a.sizing
                .as_ref()
                .filter(|s| s.capacity == capacity)
                .map(|s| (a, s))
        })
        .try_fold(0u64, |total, (a, s)| {
            if units > s.maximum {
                return Err(invalid("cache capacity exceeds its virtual extent"));
            }
            let size = units
                .checked_mul(s.unit_bytes)
                .and_then(|n| n.checked_add(a.alignment as u64 - 1))
                .ok_or_else(|| invalid("cache allocation size overflow"))?
                / a.alignment as u64
                * a.alignment as u64;
            total
                .checked_add(size)
                .ok_or_else(|| invalid("cache allocation size overflow"))
        })
}

pub(super) fn capacities(
    packages: &[Package],
    available: &[u64],
    active: u64,
    width: u64,
    page_tokens: u64,
) -> Result<(u64, u64, u64)> {
    if packages.len() != available.len()
        || packages.is_empty()
        || !(1..=64).contains(&active)
        || !(1..=8).contains(&width)
        || page_tokens == 0
    {
        return Err(invalid("invalid cache planning geometry"));
    }
    let execution = 1 + active * width;
    let minimum_pages = active * 8192u64.div_ceil(page_tokens);
    let mut cached = active * 8;
    let mut maximum_pages = u64::MAX;
    for (package, &budget) in packages.iter().zip(available) {
        let rows: Vec<_> = package
            .allocations
            .iter()
            .filter_map(|a| a.sizing.as_ref().filter(|s| s.capacity == 0))
            .collect();
        let pages: Vec<_> = package
            .allocations
            .iter()
            .filter_map(|a| a.sizing.as_ref().filter(|s| s.capacity == 1))
            .collect();
        if rows.len() != 1 || pages.is_empty() {
            return Err(invalid(
                "model requires continuation-row and KV-page allocations",
            ));
        }
        let fixed = bytes(package, 0, execution)?;
        if fixed > budget {
            return Err(invalid(
                "selected concurrency and verification width exceed available memory",
            ));
        }
        cached = cached.min((budget - fixed) / rows[0].unit_bytes);
        if rows[0].maximum < execution {
            return Err(invalid(
                "continuation virtual extent cannot hold execution state and prefix cache",
            ));
        }
        cached = cached.min(rows[0].maximum - execution);
        maximum_pages = maximum_pages.min(pages.iter().map(|s| s.maximum).min().unwrap());
    }
    let fits = |rows, pages| {
        packages.iter().zip(available).all(|(p, &budget)| {
            bytes(p, 0, rows)
                .and_then(|n| {
                    n.checked_add(bytes(p, 1, pages)?)
                        .ok_or_else(|| invalid("cache footprint overflow"))
                })
                .is_ok_and(|n| n <= budget)
        })
    };
    if !fits(execution, minimum_pages) {
        return Err(invalid(
            "available memory cannot hold execution state and minimum KV capacity",
        ));
    }
    let (mut low, mut high) = (minimum_pages, maximum_pages);
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if fits(execution, mid) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    // Snapshot virtual rows alias KV backing; they do not consume another physical budget.
    for package in packages {
        let row = package
            .allocations
            .iter()
            .find(|a| a.sizing.as_ref().is_some_and(|s| s.capacity == 0))
            .unwrap();
        let unit = row.sizing.as_ref().unwrap().unit_bytes;
        let planes: Vec<_> = package
            .allocations
            .iter()
            .filter_map(|a| {
                a.sizing
                    .as_ref()
                    .filter(|s| s.capacity == 1)
                    .map(|s| (a, s))
            })
            .collect();
        let largest = planes.iter().map(|(_, s)| s.unit_bytes).max().unwrap();
        let page: u64 = planes.iter().map(|(_, s)| s.unit_bytes).sum();
        let borrowed: u64 = planes
            .iter()
            .filter(|(a, s)| {
                super::cache_memory::alias_plane(
                    s.unit_bytes,
                    largest,
                    unit,
                    page,
                    a.alignment as u64,
                )
            })
            .map(|(a, s)| (low * s.unit_bytes).div_ceil(a.alignment as u64) * a.alignment as u64)
            .sum();
        let mapped = bytes(package, 0, execution)? + borrowed;
        cached = cached.min(mapped / unit - execution);
        while cached > 0 && bytes(package, 0, execution + cached)? > mapped {
            cached -= 1;
        }
    }
    Ok((execution + cached, low, cached))
}

impl NativeModel {
    pub(super) unsafe fn allocate_caches(
        &mut self,
        packages: &mut [Package],
        progress: &StartupProgress,
        active: usize,
        agree: super::model::CacheAgreement<'_>,
        offload_bytes: u64,
    ) -> Result<()> {
        // The host prefix store registers and streams after the caches are backed.
        let reserve = headroom(&self.server_config)?
            .checked_add(offload_bytes)
            .ok_or_else(|| invalid("prefix cache headroom overflows the device budget"))?;
        let page_tokens = self.server_config["page_tokens"]
            .as_u64()
            .ok_or_else(|| invalid("model requires physical page geometry"))?;
        let contract: crate::model::PreparedContract =
            serde_json::from_value(self.server_config["model_contract"].clone())
                .map_err(|error| invalid(format!("invalid prepared storage contract: {error}")))?;
        let width = packages[0]
            .execution
            .as_ref()
            .unwrap()
            .graphs
            .iter()
            .map(|graph| {
                contract
                    .graph(&graph.name)
                    .map(|g| g.state_rows_per_request as u64)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(1);
        let mut available = Vec::new();
        for rank in &self.ranks {
            let (free, total) = unsafe {
                (
                    rank.available_memory()? as u64,
                    if rank.integrated {
                        rank.total_memory()? as u64
                    } else {
                        0
                    },
                )
            };
            available.push(budget(free, total, reserve, rank.integrated));
        }
        let (local_rows, local_pages, _) =
            capacities(packages, &available, active as u64, width, page_tokens)?;
        // Fewer pages only shrink KV, and fewer rows only the snapshot slots aliased into it, so
        // the smallest capacity any rank can back fits every rank. Snapshot rows need no
        // dedicated backing; the alias layout below rejects rows the agreed pages cannot back.
        let (rows, pages) = agree((local_rows, local_pages))?;
        let execution = 1 + active as u64 * width;
        if rows > local_rows || pages > local_pages || rows < execution {
            return Err(invalid(
                "ranks agreed on a cache capacity this rank cannot back",
            ));
        }
        let snapshots = rows - execution;
        // Scheduler page ids count the model contract's tokens per page, which
        // exceeds the physical page when pages are sharded across ranks.
        let logical_page_tokens = contract.cache.page_tokens as u64;
        let mut geometry = packages
            .iter()
            .map(|package| cache_geometry(package, logical_page_tokens, rows, pages))
            .collect::<Result<Vec<_>>>()?;
        let layouts = packages
            .iter()
            .zip(&geometry)
            .map(|(package, geometry)| {
                let alignment = |name: &str| {
                    package
                        .allocations
                        .iter()
                        .find(|a| a.name == name)
                        .unwrap()
                        .alignment as u64
                };
                super::cache_memory::Layout::new(
                    geometry,
                    rows - snapshots,
                    &geometry
                        .planes
                        .iter()
                        .map(|p| alignment(&p.name))
                        .collect::<Vec<_>>(),
                    alignment(&geometry.row.name),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let mut aliases = vec![u32::MAX..0; snapshots as usize];
        for layout in &layouts {
            for (union, range) in aliases.iter_mut().zip(&layout.aliases) {
                if !range.is_empty() {
                    union.start = union.start.min(range.start);
                    union.end = union.end.max(range.end);
                }
            }
        }
        for range in &mut aliases {
            if range.start >= range.end {
                *range = 0..0;
            }
        }
        for geometry in &mut geometry {
            geometry.snapshot_pages = aliases.clone();
        }
        let mut gpu_bytes = 0;
        for package in packages.iter_mut() {
            gpu_bytes += bytes(package, 0, rows - snapshots)? + bytes(package, 1, pages)?;
            for allocation in &mut package.allocations {
                if let Some(sizing) = allocation.sizing.take() {
                    let units = if sizing.capacity == 0 { rows } else { pages };
                    allocation.nbytes =
                        (units * sizing.unit_bytes).div_ceil(allocation.alignment as u64) as usize
                            * allocation.alignment;
                }
            }
            for buffer in &mut package.execution.as_mut().unwrap().buffers {
                let Some(reference) = &buffer.reference else {
                    continue;
                };
                let allocation = package
                    .allocations
                    .iter()
                    .find(|a| a.name == reference.allocation)
                    .unwrap();
                if buffer.name == "state_pool" {
                    buffer.shape[0] = rows;
                    buffer.nbytes = rows * buffer.shape[1];
                } else if contract.cache.regions.iter().any(|r| {
                    r.storage.buffer == buffer.name
                        && matches!(
                            r.kind,
                            crate::model::contract::HistoryKind::PagedKeyValue
                                | crate::model::contract::HistoryKind::PagedLatent
                        )
                }) {
                    let unit = buffer.nbytes / buffer.shape[0];
                    if reference.offset != 0
                        || pages
                            .checked_mul(unit)
                            .is_none_or(|end| end > allocation.nbytes as u64)
                    {
                        return Err(invalid("paged history view exceeds physical cache backing"));
                    }
                    buffer.shape[0] = pages;
                    buffer.nbytes = pages * unit;
                }
            }
        }
        let phase = progress.phase("Allocate cache", Some(gpu_bytes));
        for (((rank, package), geometry), layout) in self
            .ranks
            .iter_mut()
            .zip(packages)
            .zip(geometry)
            .zip(layouts)
        {
            unsafe {
                rank.allocate_shared_cache(package, &geometry, &layout)?;
                rank.cache = Some(geometry);
                let binding = package
                    .execution
                    .as_ref()
                    .unwrap()
                    .buffers
                    .iter()
                    .find(|b| b.name == "memory_capacity")
                    .filter(|b| b.dtype == "int64" && b.shape == [2] && b.nbytes == 16)
                    .ok_or_else(|| invalid("model lacks runtime memory capacity binding"))?;
                let reference = binding
                    .reference
                    .as_ref()
                    .ok_or_else(|| invalid("memory capacity lacks a storage reference"))?;
                let address = rank.allocations[&reference.allocation].address + reference.offset;
                let values = [rows, pages];
                rank.driver.check(
                    "cuMemcpyHtoD_v2",
                    (rank.driver.upload)(address, values.as_ptr().cast(), 16),
                )?;
                rank.execution = package.execution.clone();
            }
        }
        self.server_config["state_rows"] = (rows - 1).into();
        self.server_config["snapshots"] = snapshots.into();
        self.server_config["pages"] = pages.into();
        phase.finish(format!(
            "{:.1} GiB · {} KV tokens before checkpoint retention · {snapshots} shared snapshot slots",
            gpu_bytes as f64 / (1u64 << 30) as f64,
            pages * logical_page_tokens
        ));
        Ok(())
    }
}

/// The row plane and page planes of one rank in name order, read before
/// the sizing records are consumed. Peer allocations never carry sizing.
fn cache_geometry(
    package: &Package,
    page_tokens: u64,
    rows: u64,
    pages: u64,
) -> Result<CacheGeometry> {
    let mut row = None;
    let mut planes = Vec::new();
    for allocation in &package.allocations {
        let Some(sizing) = allocation.sizing.as_ref() else {
            continue;
        };
        let plane = Plane {
            name: allocation.name.clone(),
            address: allocation
                .address
                .ok_or_else(|| invalid("runtime-sized allocation lacks an address"))?,
            unit_bytes: sizing.unit_bytes,
        };
        if sizing.capacity == 0 {
            if row.replace(plane).is_some() {
                return Err(invalid(
                    "rank declares more than one continuation-row allocation",
                ));
            }
        } else {
            planes.push(plane);
        }
    }
    planes.sort_unstable_by(|a, b| a.name.cmp(&b.name));
    let geometry = CacheGeometry {
        snapshot_pages: Vec::new(),
        page_tokens,
        page_capacity: pages,
        row_capacity: rows,
        row: row.ok_or_else(|| invalid("rank lacks a continuation-row allocation"))?,
        planes,
    };
    geometry.validate()?;
    Ok(geometry)
}

#[cfg(test)]
mod tests {
    use super::super::{
        format::{Allocation, Context, Memory},
        protobuf::wire::Sizing,
    };
    use super::*;

    #[test]
    fn integrated_memory_includes_reclaimable_ram_but_not_swap() {
        let info = "MemFree: 1024 kB\nMemAvailable: 8388608 kB\nSwapFree: 16777216 kB\n";
        assert_eq!(shared_available(info, 16 << 30).unwrap(), 8 << 30);
        assert_eq!(shared_available(info, 4 << 30).unwrap(), 4 << 30);
        for invalid in [
            "MemFree: 1024 kB\nSwapFree: 16777216 kB\n",
            "MemAvailable: -1 kB",
            "MemAvailable: 1024 MB",
            "MemAvailable: 18446744073709551615 kB",
        ] {
            assert!(shared_available(invalid, usize::MAX).is_err());
        }
    }

    #[test]
    fn serving_headroom_preserves_the_user_reserve() {
        assert_eq!(headroom(&serde_json::json!({})).unwrap(), 64 << 20);
        assert_eq!(
            headroom(&serde_json::json!({"reserve_bytes": 2u64 << 30})).unwrap(),
            (2 << 30) + (64 << 20)
        );
        for reserve in [
            serde_json::json!(-1),
            serde_json::json!(1.5),
            serde_json::json!("0"),
            serde_json::json!(null),
            serde_json::json!(u64::MAX),
        ] {
            assert!(headroom(&serde_json::json!({"reserve_bytes": reserve})).is_err());
        }
    }

    #[test]
    fn integrated_gpus_keep_host_memory_for_the_server_beyond_the_reserve() {
        let gib = 1u64 << 30;
        assert_eq!(budget(40 * gib, 120 * gib, 5 * gib, false), 35 * gib);
        // Two GiB for the server and a twentieth of memory for the system.
        assert_eq!(budget(40 * gib, 120 * gib, 5 * gib, true), 27 * gib);
        assert_eq!(budget(40 * gib, 120 * gib, 0, true), 32 * gib);
        assert_eq!(budget(6 * gib, 120 * gib, 0, true), 0);
    }

    fn package(row_bytes: u64, page_bytes: u64, maximum_pages: u64) -> Package {
        Package {
            resources: std::collections::HashMap::new(),
            events: vec![],
            architecture: [12, 0],
            modules: vec![],
            functions: vec![],
            graphs: vec![],
            reservations: vec![],
            execution: None,
            context: Context::default(),
            allocations: [
                ("state", 0, row_bytes, 4096),
                ("kv", 1, page_bytes, maximum_pages),
            ]
            .into_iter()
            .map(|(name, capacity, unit_bytes, maximum)| Allocation {
                name: name.into(),
                nbytes: 0,
                alignment: 65536,
                file: None,
                data_spans: vec![],
                relocations: vec![],
                address: Some(0x200000000000),
                memory: Memory::Device,
                sizing: Some(Sizing {
                    capacity,
                    unit_bytes,
                    maximum,
                }),
            })
            .collect(),
        }
    }

    #[test]
    fn cache_geometry_orders_named_planes_independently_of_allocation_traversal() {
        let mut p = package(100 << 20, 2 << 20, 100000);
        p.allocations.push(Allocation {
            name: "kv-draft".into(),
            nbytes: 0,
            alignment: 65536,
            file: None,
            data_spans: vec![],
            relocations: vec![],
            address: Some(0x300000000000),
            memory: Memory::Device,
            sizing: Some(Sizing {
                capacity: 1,
                unit_bytes: 4096,
                maximum: 100000,
            }),
        });
        p.allocations.push(Allocation {
            name: "peer-0".into(),
            nbytes: 1 << 20,
            alignment: 65536,
            file: None,
            data_spans: vec![],
            relocations: vec![],
            address: Some(0x400000000000),
            memory: Memory::Peer,
            sizing: None,
        });
        let geometry = cache_geometry(&p, 256, 40, 5000).unwrap();
        assert_eq!(geometry.page_tokens, 256);
        assert_eq!((geometry.row_capacity, geometry.page_capacity), (40, 5000));
        assert_eq!(geometry.row.name, "state");
        assert_eq!(geometry.row.unit_bytes, 100 << 20);
        assert_eq!(
            geometry
                .planes
                .iter()
                .map(|plane| (plane.name.as_str(), plane.unit_bytes))
                .collect::<Vec<_>>(),
            [("kv", 2 << 20), ("kv-draft", 4096)]
        );
        assert_eq!(geometry.planes[1].address, 0x300000000000);
        p.allocations.reverse();
        let reordered = cache_geometry(&p, 256, 40, 5000).unwrap();
        assert_eq!(geometry, reordered);
        assert!(geometry.same_layout(&reordered));
        let original_ranks = vec![geometry.clone(); 4];
        let mut reordered_ranks = original_ranks.clone();
        reordered_ranks[3] = reordered;
        assert_eq!(
            crate::offload::geometry::fingerprint(&original_ranks),
            crate::offload::geometry::fingerprint(&reordered_ranks),
        );
        let mut options = package(4096, 4096, 16);
        assert_eq!(
            crate::offload::geometry::fingerprint(&[cache_geometry(&options, 64, 8, 16).unwrap()]),
            crate::offload::geometry::fingerprint(
                &[cache_geometry(&options, 64, 80, 160).unwrap()]
            )
        );
        options.allocations.push(Allocation {
            name: "state-2".into(),
            nbytes: 0,
            alignment: 65536,
            file: None,
            data_spans: vec![],
            relocations: vec![],
            address: Some(0x500000000000),
            memory: Memory::Device,
            sizing: Some(Sizing {
                capacity: 0,
                unit_bytes: 4096,
                maximum: 16,
            }),
        });
        assert!(cache_geometry(&options, 64, 8, 16).is_err());
    }

    #[test]
    fn lower_concurrency_and_verification_move_physical_budget_to_kv() {
        let p = [package(100 << 20, 2 << 20, 100000)];
        let budget = [80 << 30];
        let full = capacities(&p, &budget, 32, 6, 64).unwrap();
        let reduced = capacities(&p, &budget, 4, 6, 64).unwrap();
        let target_only = capacities(&p, &budget, 4, 1, 64).unwrap();
        assert!(full.0 > reduced.0 && reduced.0 > target_only.0);
        assert!(full.1 < reduced.1 && reduced.1 < target_only.1);
        assert_eq!(reduced.0, 1 + 4 * 6 + reduced.2);
        for (rows, pages, snapshots) in [full, reduced, target_only] {
            let used = bytes(&p[0], 0, rows - snapshots).unwrap() + bytes(&p[0], 1, pages).unwrap();
            assert!(used <= budget[0] && budget[0] - used < 2 << 20);
        }
        // Capture concurrency is deliberately absent from the allocation policy.
        assert_eq!(reduced, capacities(&p, &budget, 4, 6, 64).unwrap());
    }

    #[test]
    fn ranks_share_capacities_and_respect_alignment_and_virtual_extents() {
        let p = [package(123456, 45678, 1000), package(246912, 91356, 1000)];
        let (rows, pages, snapshots) = capacities(&p, &[256 << 20, 80 << 20], 4, 3, 256).unwrap();
        assert!(pages <= 1000);
        for (p, budget) in p.iter().zip([256 << 20, 80 << 20]) {
            assert!(bytes(p, 0, rows - snapshots).unwrap() + bytes(p, 1, pages).unwrap() <= budget);
        }
        assert!(capacities(&p, &[256 << 20, 1 << 20], 4, 3, 256).is_err());
        let (_, pages, _) = capacities(&[package(1024, 1024, 1024)], &[1 << 30], 1, 1, 64).unwrap();
        assert_eq!(pages, 1024);
    }
}
