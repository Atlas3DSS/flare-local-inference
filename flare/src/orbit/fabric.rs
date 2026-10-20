//! TCP package transfers over the same neighbor interfaces as the RoCE ring.

use std::net::{Ipv4Addr, SocketAddrV4};

use flare_helios::execution::roce_fabric;
use serde::{Deserialize, Serialize};

use super::{
    Failure,
    grant::{Grant, Member},
    store::Node,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Link {
    pub from: SocketAddrV4,
    pub to: SocketAddrV4,
}

pub fn discover() -> Result<Vec<u8>, Failure> {
    #[cfg(target_os = "linux")]
    {
        let fabric = roce_fabric::discover().map_err(|e| e.to_string())?;
        Ok(roce_fabric::encode(&fabric.endpoints))
    }
    #[cfg(not(target_os = "linux"))]
    Err("fabric package transfers require Linux agents with active RoCE interfaces".into())
}

/// Plan each clockwise edge using the collective transport's subnet/device matching.
pub fn plan(nodes: &[Node], records: &[Vec<u8>]) -> Result<Vec<Vec<Link>>, Failure> {
    if nodes.len() != records.len() || !(2..=16).contains(&nodes.len()) {
        return Err("a transfer ring needs 2..16 nodes and their fabric endpoints".into());
    }
    let endpoints = records
        .iter()
        .map(|r| roce_fabric::decode(r))
        .collect::<flare_helios::Result<Vec<_>>>()
        .map_err(|e| e.to_string())?;
    let rails = endpoints
        .iter()
        .map(Vec::len)
        .min()
        .unwrap()
        .min(roce_fabric::MAX_RAILS);
    let address = |rank: usize, device: u32| -> Result<SocketAddrV4, Failure> {
        let ip = endpoints[rank][device as usize]
            .ipv4
            .ok_or("fabric TCP requires an IPv4 address and subnet for every selected link")?
            .0;
        let port = super::with_port(&nodes[rank].address)
            .rsplit_once(':')
            .ok_or("agent address lacks a port")?
            .1
            .parse()?;
        Ok(SocketAddrV4::new(ip, port))
    };
    (0..nodes.len())
        .map(|rank| {
            let next = (rank + 1) % nodes.len();
            let routes = roce_fabric::plan_ring_routes(&endpoints, rank, rails)
                .map_err(|e| e.to_string())?;
            routes[next]
                .iter()
                .map(|&(local, remote)| {
                    Ok(Link {
                        from: address(rank, local)?,
                        to: address(next, remote)?,
                    })
                })
                .collect()
        })
        .collect()
}

/// The payload path, with clockwise ties, identical to collective ring routing.
pub fn path(world: usize, source: usize, destination: usize) -> Result<Vec<usize>, Failure> {
    if !(2..=16).contains(&world)
        || source >= world
        || destination >= world
        || source == destination
    {
        return Err("invalid fabric transfer ranks".into());
    }
    let clockwise = (destination + world - source) % world;
    let step = if clockwise <= world - clockwise {
        1
    } else {
        world - 1
    };
    let mut path = vec![source];
    while *path.last().unwrap() != destination {
        path.push((path.last().unwrap() + step) % world);
    }
    Ok(path)
}

pub fn validate(grant: &Grant) -> Result<(), Failure> {
    if !grant.roce_ring {
        return Err("ring transfer lacks a signed fabric topology".into());
    }
    validate_ring(&grant.members, &grant.transfer_ring)
}

pub fn validate_ring(members: &[Member], ring: &[Vec<Link>]) -> Result<(), Failure> {
    let world = members.len();
    if !(2..=16).contains(&world) || ring.len() != world {
        return Err("ring transfer lacks a signed fabric topology".into());
    }
    for (rank, member) in members.iter().enumerate() {
        if member.rank != rank
            || members[..rank]
                .iter()
                .any(|m| m.public_key == member.public_key)
        {
            return Err("transfer ring members must have distinct keys in rank order".into());
        }
        let links = &ring[rank];
        if links.is_empty() || links.len() > roce_fabric::MAX_RAILS {
            return Err("ring edge lacks fabric TCP links".into());
        }
        for endpoint in links.iter().flat_map(|l| [l.from, l.to]) {
            if endpoint.port() == 0
                || endpoint.ip().is_unspecified()
                || endpoint.ip().is_multicast()
            {
                return Err("invalid fabric TCP endpoint".into());
            }
        }
    }
    Ok(())
}

/// Source within the ring; an outside source first stages a complete copy on rank zero.
pub fn source_rank(grant: &Grant) -> usize {
    grant
        .package
        .source
        .as_ref()
        .and_then(|s| {
            grant
                .members
                .iter()
                .position(|m| m.public_key == s.public_key)
        })
        .unwrap_or(0)
}

/// Local bind address and remote listener for a connection to an adjacent member.
pub fn hop(
    grant: &Grant,
    from: usize,
    to: usize,
    rail: usize,
) -> Result<(Ipv4Addr, SocketAddrV4), Failure> {
    validate(grant)?;
    ring_hop(&grant.members, &grant.transfer_ring, from, to, rail)
}

pub fn ring_hop(
    members: &[Member],
    ring: &[Vec<Link>],
    from: usize,
    to: usize,
    rail: usize,
) -> Result<(Ipv4Addr, SocketAddrV4), Failure> {
    validate_ring(members, ring)?;
    let world = members.len();
    if from >= world || to >= world {
        return Err("fabric hop is outside the ring".into());
    }
    let (edge, forward) = if to == (from + 1) % world {
        (from, true)
    } else if from == (to + 1) % world {
        (to, false)
    } else {
        return Err("fabric TCP connections must join ring neighbors".into());
    };
    let links = &ring[edge];
    // A two-rank ring can reach its one neighbor in either direction using the same links.
    let link = &links[rail % links.len()];
    Ok(if forward {
        (*link.from.ip(), link.to)
    } else {
        (*link.to.ip(), link.from)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_take_shortest_direction_and_clockwise_ties() {
        for world in 2..=16 {
            for source in 0..world {
                for destination in 0..world {
                    if source == destination {
                        continue;
                    }
                    let route = path(world, source, destination).unwrap();
                    let clockwise = (destination + world - source) % world;
                    assert_eq!(route.len() - 1, clockwise.min(world - clockwise));
                    assert_eq!(route[0], source);
                    assert_eq!(*route.last().unwrap(), destination);
                    for edge in route.windows(2) {
                        assert_eq!(
                            (edge[1] + world - edge[0]) % world,
                            if clockwise <= world - clockwise {
                                1
                            } else {
                                world - 1
                            }
                        );
                    }
                }
            }
        }
        assert_eq!(path(4, 0, 2).unwrap(), [0, 1, 2]);
        assert_eq!(path(4, 0, 3).unwrap(), [0, 3]);
        assert!(path(4, 4, 0).is_err());
        assert!(path(4, 0, 0).is_err());
    }

    #[test]
    fn physical_ring_selects_only_neighbor_subnets_on_both_rails() {
        let nodes = (0..4)
            .map(|rank| Node {
                name: format!("rank{rank}"),
                fingerprint: String::new(),
                public_key: rank.to_string(),
                address: format!("10.137.203.{}:29900", rank + 1),
            })
            .collect::<Vec<_>>();
        let records = [
            "a 10.10.4.1/24\nb 10.10.0.2/24\nc 10.10.5.1/24\nd 10.10.1.2/24\n",
            "a 10.10.6.1/24\nb 10.10.4.2/24\nc 10.10.7.1/24\nd 10.10.5.2/24\n",
            "a 10.10.2.2/24\nb 10.10.6.2/24\nc 10.10.3.2/24\nd 10.10.7.2/24\n",
            "a 10.10.0.1/24\nb 10.10.2.1/24\nc 10.10.1.1/24\nd 10.10.3.1/24\n",
        ]
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect::<Vec<_>>();
        let ring = plan(&nodes, &records).unwrap();
        for (edge, subnets) in ring.iter().zip([[4, 5], [6, 7], [2, 3], [0, 1]]) {
            assert_eq!(edge.len(), 2);
            for (link, subnet) in edge.iter().zip(subnets) {
                assert_eq!(link.from.ip().octets()[..3], [10, 10, subnet]);
                assert_eq!(link.to.ip().octets()[..3], [10, 10, subnet]);
                assert_ne!(link.from.ip(), link.to.ip());
            }
        }
        let mut broken = records.clone();
        broken[2] = b"a 192.168.0.1/24\n".to_vec();
        assert!(plan(&nodes, &broken).is_err());
    }
}
