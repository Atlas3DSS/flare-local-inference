use std::collections::{HashMap, HashSet};
use std::path::{Component, Path};

use crate::{Error, ExecutionError, Result};

pub(super) fn invalid(message: impl Into<String>) -> Error {
    Error::execution(ExecutionError::Manifest, message)
}

#[derive(Debug)]
pub(super) struct Package {
    pub version: u32,
    pub architecture: [i32; 2],
    pub allocations: Vec<Allocation>,
    pub modules: Vec<Module>,
    pub functions: Vec<Function>,
    pub graphs: Vec<Graph>,
    pub reservations: Vec<Reservation>,
    pub execution: Option<super::protobuf::wire::Execution>,
    pub context: Context,
}

#[derive(Debug, Default)]
pub(super) struct Context {
    pub flags: u32,
    pub cache: i32,
    pub limits: Vec<[usize; 2]>,
}

#[derive(Debug)]
pub(super) struct Reservation {
    pub address: u64,
    pub nbytes: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) enum Memory {
    #[default]
    Device,
    MappedHost,
}

#[derive(Debug)]
pub(super) struct Allocation {
    pub name: String,
    pub nbytes: usize,
    pub alignment: usize,
    pub file: Option<String>,
    pub data_spans: Vec<[usize; 2]>,
    pub relocations: Vec<Relocation>,
    pub address: Option<u64>,
    pub memory: Memory,
}

impl Allocation {
    pub fn spans(&self) -> impl Iterator<Item = [usize; 2]> + '_ {
        self.data_spans.iter().copied()
    }

    pub fn stored_bytes(&self) -> usize {
        self.spans().map(|span| span[1]).sum()
    }
}

#[derive(Debug)]
pub(super) struct Module {
    pub name: String,
    pub file: String,
    pub globals: Vec<Global>,
}

#[derive(Debug)]
pub(super) struct Global {
    pub symbol: String,
    pub file: String,
    pub nbytes: usize,
    pub relocations: Vec<Relocation>,
}

#[derive(Debug)]
pub(super) struct Function {
    pub module: String,
    pub symbol: String,
    pub parameter_sizes: Vec<usize>,
    pub attributes: Vec<[i32; 2]>,
}

#[derive(Debug)]
pub(super) struct Reference {
    pub allocation: String,
    pub offset: usize,
}

#[derive(Debug)]
pub(super) enum RelocationKind {
    Pointer,
    TensorMap,
}

#[derive(Debug)]
pub(super) struct Relocation {
    pub offset: usize,
    pub target: Reference,
    pub kind: RelocationKind,
}

impl Relocation {
    pub fn width(&self) -> usize {
        match self.kind {
            RelocationKind::Pointer => 8,
            RelocationKind::TensorMap => 128,
        }
    }
}

#[derive(Debug)]
pub(super) struct Blob {
    pub data: Vec<u8>,
    pub relocations: Vec<Relocation>,
}

#[derive(Debug)]
pub(super) struct Attribute {
    pub id: i32,
    pub value: Blob,
}

#[derive(Debug)]
pub(super) enum Node {
    Kernel {
        function: usize,
        grid: [u32; 3],
        block: [u32; 3],
        shared: u32,
        arguments: Vec<Blob>,
        attributes: Vec<Attribute>,
    },
    Memset {
        destination: Reference,
        pitch: usize,
        value: u32,
        element_size: u32,
        width: usize,
        height: usize,
    },
    Copy {
        source: Reference,
        destination: Reference,
        nbytes: usize,
    },
    Empty,
}

#[derive(Debug)]
pub(super) struct Edge {
    pub source: usize,
    pub destination: usize,
    pub data: [u8; 8],
}

#[derive(Debug)]
pub(super) struct Graph {
    pub name: String,
    pub flags: u64,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

fn filename(value: &str) -> Result<()> {
    let mut components = Path::new(value).components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(invalid(
            "package data files must be direct children of the package directory",
        ));
    }
    Ok(())
}

impl Package {
    pub fn validate(&self) -> Result<()> {
        if self.version != 3 || self.allocations.is_empty() || self.graphs.is_empty() {
            return Err(invalid("unsupported or empty runtime package"));
        }
        if self
            .execution
            .as_ref()
            .is_some_and(|execution| !matches!(execution.abi_version, 2 | 3))
        {
            return Err(invalid("unsupported execution contract version"));
        }
        if self.context.limits.iter().any(|limit| limit[0] > 6)
            || !(0..=3).contains(&self.context.cache)
        {
            return Err(invalid("unsupported CUDA context configuration"));
        }
        let mut allocations = HashMap::new();
        let mut addresses = Vec::new();
        let mut reservations = Vec::new();
        for r in &self.reservations {
            if r.address == 0 || r.nbytes == 0 || r.address.checked_add(r.nbytes as u64).is_none() {
                return Err(invalid("invalid device address reservation"));
            }
            reservations.push((r.address, r.address + r.nbytes as u64));
        }
        reservations.sort_unstable();
        if reservations.windows(2).any(|r| r[0].1 > r[1].0) {
            return Err(invalid("device address reservations overlap"));
        }
        for a in &self.allocations {
            let mut end = 0;
            for [offset, size] in a.spans() {
                if size == 0
                    || offset < end
                    || offset.checked_add(size).is_none_or(|n| n > a.nbytes)
                {
                    return Err(invalid(
                        "allocation data spans must be ordered, disjoint, and in bounds",
                    ));
                }
                end = offset + size;
            }
            if a.file.is_some() != (a.stored_bytes() > 0) {
                return Err(invalid("allocation payload file must match its data spans"));
            }
            if matches!(a.memory, Memory::MappedHost)
                && (a.nbytes % 4096 != 0 || a.address.is_none_or(|v| v % 4096 != 0))
            {
                return Err(invalid(
                    "mapped host allocations must cover complete host pages",
                ));
            }
            if a.name.is_empty()
                || a.nbytes == 0
                || !a.alignment.is_power_of_two()
                || a.nbytes.checked_add(a.alignment - 1).is_none()
                || allocations.insert(a.name.as_str(), a.nbytes).is_some()
            {
                return Err(invalid("invalid or duplicate runtime allocation"));
            }
            if let Some(address) = a.address {
                let end = address
                    .checked_add(a.nbytes as u64)
                    .ok_or_else(|| invalid("allocation address overflow"))?;
                if address == 0 || address % a.alignment as u64 != 0 {
                    return Err(invalid("invalid recorded allocation address"));
                }
                if matches!(a.memory, Memory::Device)
                    && !self
                        .reservations
                        .iter()
                        .any(|r| r.address <= address && end <= r.address + r.nbytes as u64)
                {
                    return Err(invalid(
                        "recorded device allocation lies outside its reservation",
                    ));
                }
                addresses.push((address, end));
            } else if matches!(a.memory, Memory::MappedHost) {
                return Err(invalid(
                    "mapped host allocation requires a recorded address",
                ));
            }
            if let Some(file) = &a.file {
                filename(file)?;
            } else if a
                .relocations
                .iter()
                .any(|r| matches!(r.kind, RelocationKind::TensorMap))
            {
                return Err(invalid(
                    "tensor-map storage requires an initialized descriptor",
                ));
            }
        }
        addresses.sort_unstable();
        if addresses.windows(2).any(|a| a[0].1 > a[1].0) {
            return Err(invalid("recorded allocation addresses overlap"));
        }
        let reference = |r: &Reference, size: usize| -> Result<()> {
            if !allocations
                .get(r.allocation.as_str())
                .is_some_and(|n| r.offset.checked_add(size).is_some_and(|end| end <= *n))
            {
                return Err(invalid(format!(
                    "reference exceeds allocation {:?}",
                    r.allocation
                )));
            }
            Ok(())
        };
        let relocations = |size: usize, values: &[Relocation]| -> Result<()> {
            let mut fields: Vec<_> = values.iter().collect();
            fields.sort_by_key(|r| r.offset);
            let mut end = 0;
            for r in fields {
                if r.offset < end || r.offset.checked_add(r.width()).is_none_or(|v| v > size) {
                    return Err(invalid("invalid or overlapping relocation fields"));
                }
                reference(&r.target, 1)?;
                end = r.offset + r.width();
            }
            Ok(())
        };
        for a in &self.allocations {
            relocations(a.nbytes, &a.relocations)?;
        }
        let mut modules = HashSet::new();
        for module in &self.modules {
            filename(&module.file)?;
            if module.name.is_empty() || !modules.insert(&module.name) {
                return Err(invalid("invalid or duplicate module name"));
            }
            let mut globals = HashSet::new();
            for global in &module.globals {
                filename(&global.file)?;
                if global.symbol.is_empty()
                    || global.symbol.contains('\0')
                    || global.nbytes == 0
                    || !globals.insert(&global.symbol)
                {
                    return Err(invalid("invalid or duplicate module global"));
                }
                relocations(global.nbytes, &global.relocations)?;
            }
        }
        for function in &self.functions {
            if !modules.contains(&function.module)
                || function.symbol.is_empty()
                || function.symbol.contains('\0')
                || function.parameter_sizes.contains(&0)
                || function
                    .attributes
                    .iter()
                    .any(|a| ![8, 9, 14, 15].contains(&a[0]))
            {
                return Err(invalid("invalid runtime kernel definition"));
            }
        }
        let mut names = HashSet::new();
        for graph in &self.graphs {
            if graph.name.is_empty()
                || !names.insert(&graph.name)
                || graph.nodes.is_empty()
                || graph.flags & !9 != 0
            {
                return Err(invalid("invalid or duplicate graph name"));
            }
            for node in &graph.nodes {
                match node {
                    Node::Kernel {
                        function,
                        grid,
                        block,
                        arguments,
                        attributes,
                        ..
                    } => {
                        let f = self
                            .functions
                            .get(*function)
                            .ok_or_else(|| invalid("unknown kernel"))?;
                        if grid.contains(&0)
                            || block.contains(&0)
                            || arguments
                                .iter()
                                .map(|a| a.data.len())
                                .ne(f.parameter_sizes.iter().copied())
                        {
                            return Err(invalid(
                                "kernel argument layout or launch dimensions are invalid",
                            ));
                        }
                        for a in arguments {
                            relocations(a.data.len(), &a.relocations)?;
                        }
                        let mut ids = HashSet::new();
                        for a in attributes {
                            if ![1, 2, 4, 5, 8, 9, 10, 11, 14].contains(&a.id)
                                || a.value.data.len() != 64
                                || !ids.insert(a.id)
                            {
                                return Err(invalid(
                                    "unsupported or duplicate kernel-node attribute",
                                ));
                            }
                            relocations(64, &a.value.relocations)?;
                        }
                    }
                    Node::Memset {
                        destination,
                        pitch,
                        element_size,
                        width,
                        height,
                        ..
                    } => {
                        let row = width
                            .checked_mul(*element_size as usize)
                            .ok_or_else(|| invalid("memset extent overflow"))?;
                        if ![1, 2, 4].contains(element_size)
                            || row == 0
                            || *height == 0
                            || (*height > 1 && *pitch < row)
                        {
                            return Err(invalid("invalid memset dimensions"));
                        }
                        let extent = (height - 1)
                            .checked_mul(*pitch)
                            .and_then(|v| v.checked_add(row))
                            .ok_or_else(|| invalid("memset extent overflow"))?;
                        reference(destination, extent)?;
                    }
                    Node::Copy {
                        source,
                        destination,
                        nbytes,
                    } => {
                        if *nbytes == 0 {
                            return Err(invalid("empty graph copy"));
                        }
                        reference(source, *nbytes)?;
                        reference(destination, *nbytes)?;
                    }
                    Node::Empty => {}
                }
            }
            let mut seen = HashSet::new();
            let mut indegree = vec![0usize; graph.nodes.len()];
            let mut successors = vec![vec![]; graph.nodes.len()];
            for edge in &graph.edges {
                if edge.source >= graph.nodes.len()
                    || edge.destination >= graph.nodes.len()
                    || edge.data[0] > 2
                    || edge.data[1] != 0
                    || edge.data[2] > 1
                    || edge.data[3..].iter().any(|v| *v != 0)
                    || !seen.insert((edge.source, edge.destination))
                {
                    return Err(invalid("invalid graph dependency"));
                }
                indegree[edge.destination] += 1;
                successors[edge.source].push(edge.destination);
            }
            let mut ready: Vec<_> = indegree
                .iter()
                .enumerate()
                .filter_map(|(i, n)| (*n == 0).then_some(i))
                .collect();
            let mut visited = 0;
            while let Some(node) = ready.pop() {
                visited += 1;
                for &next in &successors[node] {
                    indegree[next] -= 1;
                    if indegree[next] == 0 {
                        ready.push(next);
                    }
                }
            }
            if visited != graph.nodes.len() {
                return Err(invalid("graph dependency cycle"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn package() -> Package {
        Package {
            version: 3,
            architecture: [12, 0],
            allocations: vec![Allocation {
                name: "state".into(),
                nbytes: 4096,
                alignment: 256,
                file: Some("state.bin".into()),
                data_spans: vec![[0, 4096]],
                relocations: vec![],
                address: None,
                memory: Memory::Device,
            }],
            modules: vec![],
            functions: vec![],
            reservations: vec![],
            execution: None,
            context: Context::default(),
            graphs: vec![Graph {
                name: "copy".into(),
                flags: 0,
                nodes: vec![Node::Empty, Node::Empty],
                edges: vec![],
            }],
        }
    }

    #[test]
    fn recorded_device_spans_require_a_containing_reservation() {
        let mut value = package();
        value.allocations[0].address = Some(0x200000000000);
        assert!(value.validate().is_err());
        value.reservations.push(Reservation {
            address: 0x200000000000,
            nbytes: 8192,
        });
        assert!(value.validate().is_ok());
        value.allocations[0].nbytes = 8193;
        assert!(value.validate().is_err());
    }

    #[test]
    fn packed_payload_spans_are_bounded_and_match_the_file() {
        let mut value = package();
        for spans in [
            vec![[0, 0]],
            vec![[4095, 2]],
            vec![[16, 32], [32, 16]],
            vec![],
        ] {
            value.allocations[0].data_spans = spans;
            assert!(value.validate().is_err());
        }
        value.allocations[0].data_spans = vec![[16, 32], [128, 64]];
        assert!(value.validate().is_ok());
        assert_eq!(value.allocations[0].stored_bytes(), 96);
        value.allocations[0].file = None;
        assert!(value.validate().is_err());
        value.allocations[0].data_spans.clear();
        assert!(value.validate().is_ok());
    }

    #[test]
    fn pointer_fields_cannot_overlap_or_reference_an_allocation_end() {
        let mut value = package();
        let relocation = || Relocation {
            offset: 0,
            kind: RelocationKind::Pointer,
            target: Reference {
                allocation: "state".into(),
                offset: 4095,
            },
        };
        value.allocations[0].relocations.push(relocation());
        assert!(value.validate().is_ok());
        value.allocations[0].relocations[0].target.offset = 4096;
        assert!(value.validate().is_err());
        value.allocations[0].relocations = vec![relocation(), relocation()];
        assert!(value.validate().is_err());
    }

    #[test]
    fn graph_cycles_are_rejected() {
        let mut value = package();
        value.graphs[0].edges = vec![
            Edge {
                source: 0,
                destination: 1,
                data: [0; 8],
            },
            Edge {
                source: 1,
                destination: 0,
                data: [0; 8],
            },
        ];
        assert!(value.validate().is_err());
    }

    #[test]
    fn data_paths_cannot_escape_the_package() {
        let mut value = package();
        value.allocations[0].file = Some("../weights.bin".into());
        assert!(value.validate().is_err());
    }

    #[test]
    fn mapped_host_extents_cover_complete_pages() {
        let mut value = package();
        value.allocations[0].memory = Memory::MappedHost;
        value.allocations[0].address = Some(0x100000000000);
        assert!(value.validate().is_ok());
        value.allocations[0].nbytes = 4000;
        value.allocations[0].data_spans = vec![[0, 4000]];
        assert!(value.validate().is_err());
    }
}
