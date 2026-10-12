//! Choose physical cache capacities within reserved virtual address ranges.

use super::{
    NativeModel,
    format::{Package, invalid},
};
use crate::{Result, startup::StartupProgress};

// Graphs are uploaded before sizing caches; leave room for driver bookkeeping
// and the execution streams and events created when the server starts.
const HEADROOM_BYTES: u64 = 64 << 20;

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
    let admission = active * 3;
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
        let fixed = bytes(package, 0, execution + admission)?;
        if fixed > budget {
            return Err(invalid(
                "selected concurrency and verification width exceed available memory",
            ));
        }
        cached = cached.min(((budget - fixed) / 4 / rows[0].unit_bytes).max(8));
        if rows[0].maximum < execution + admission + 8 {
            return Err(invalid(
                "continuation virtual extent cannot hold execution state and prefix cache",
            ));
        }
        cached = cached.min(rows[0].maximum - execution - admission);
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
    loop {
        let rows = execution + admission + cached;
        if fits(rows, minimum_pages) {
            break;
        }
        if cached <= 8 {
            return Err(invalid(
                "available memory cannot hold execution state, prefix cache, and minimum KV capacity",
            ));
        }
        cached -= 1;
    }
    let rows = execution + admission + cached;
    let (mut low, mut high) = (minimum_pages, maximum_pages);
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if fits(rows, mid) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    Ok((rows, low, admission + cached))
}

impl NativeModel {
    pub(super) unsafe fn allocate_caches(
        &mut self,
        packages: &mut [Package],
        progress: &StartupProgress,
        active: usize,
    ) -> Result<()> {
        let reserve = headroom(&self.server_config)?;
        let page_tokens = self.server_config["page_tokens"]
            .as_u64()
            .ok_or_else(|| invalid("model requires physical page geometry"))?;
        let width = packages[0]
            .execution
            .as_ref()
            .unwrap()
            .graphs
            .iter()
            .filter(|g| g.kind == "speculative")
            .map(|g| g.token_capacity)
            .max()
            .unwrap_or(1);
        let mut available = Vec::new();
        for rank in &self.ranks {
            let (mut free, mut total) = (0, 0);
            unsafe {
                rank.driver
                    .check("cuCtxSetCurrent", (rank.driver.context_set)(rank.context))?;
                rank.driver.check(
                    "cuMemGetInfo_v2",
                    (rank.driver.memory_info)(&mut free, &mut total),
                )?;
            }
            available.push((free as u64).saturating_sub(reserve));
        }
        let (rows, pages, snapshots) =
            capacities(packages, &available, active as u64, width, page_tokens)?;
        let mut gpu_bytes = 0;
        for package in packages.iter_mut() {
            for allocation in &mut package.allocations {
                if let Some(sizing) = allocation.sizing.take() {
                    let units = if sizing.capacity == 0 { rows } else { pages };
                    allocation.nbytes =
                        (units * sizing.unit_bytes).div_ceil(allocation.alignment as u64) as usize
                            * allocation.alignment;
                    gpu_bytes += allocation.nbytes as u64;
                }
            }
            let pool = package
                .execution
                .as_mut()
                .unwrap()
                .buffers
                .iter_mut()
                .find(|b| b.name == "state_pool")
                .ok_or_else(|| invalid("model lacks continuation storage binding"))?;
            pool.shape[0] = rows;
            pool.nbytes = rows * pool.shape[1];
        }
        let phase = progress.phase("Allocate cache", Some(gpu_bytes));
        for (rank, package) in self.ranks.iter_mut().zip(packages) {
            unsafe {
                rank.allocate(package, &phase)?;
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
            "{:.1} GiB · {} cached tokens · {snapshots} state rows",
            gpu_bytes as f64 / (1u64 << 30) as f64,
            pages * page_tokens
        ));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        format::{Allocation, Context, Memory},
        protobuf::wire::Sizing,
    };
    use super::*;

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

    fn package(row_bytes: u64, page_bytes: u64, maximum_pages: u64) -> Package {
        Package {
            resources: std::collections::HashMap::new(),
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
    fn lower_concurrency_and_verification_move_physical_budget_to_kv() {
        let p = [package(100 << 20, 2 << 20, 100000)];
        let budget = [80 << 30];
        let full = capacities(&p, &budget, 32, 6, 64).unwrap();
        let reduced = capacities(&p, &budget, 4, 6, 64).unwrap();
        let target_only = capacities(&p, &budget, 4, 1, 64).unwrap();
        assert!(full.0 > reduced.0 && reduced.0 > target_only.0);
        assert!(full.1 < reduced.1 && reduced.1 < target_only.1);
        assert_eq!(reduced.0, 1 + 4 * 6 + reduced.2);
        for (rows, pages, _) in [full, reduced, target_only] {
            let used = bytes(&p[0], 0, rows).unwrap() + bytes(&p[0], 1, pages).unwrap();
            assert!(used <= budget[0] && budget[0] - used < 2 << 20);
        }
        // Capture concurrency is deliberately absent from the allocation policy.
        assert_eq!(reduced, capacities(&p, &budget, 4, 6, 64).unwrap());
    }

    #[test]
    fn ranks_share_capacities_and_respect_alignment_and_virtual_extents() {
        let p = [package(123456, 45678, 1000), package(246912, 91356, 1000)];
        let (rows, pages, _) = capacities(&p, &[256 << 20, 80 << 20], 4, 3, 256).unwrap();
        assert!(pages <= 1000);
        for (p, budget) in p.iter().zip([256 << 20, 80 << 20]) {
            assert!(bytes(p, 0, rows).unwrap() + bytes(p, 1, pages).unwrap() <= budget);
        }
        assert!(capacities(&p, &[256 << 20, 1 << 20], 4, 3, 256).is_err());
        let (_, pages, _) = capacities(&[package(1024, 1024, 1024)], &[1 << 30], 1, 1, 64).unwrap();
        assert_eq!(pages, 1024);
    }
}
