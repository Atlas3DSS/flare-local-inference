//! Fixture publisher shared by native package tests; never an official release key.

use super::{AuthorizedPackage, Publisher, VerifiedPackage};
use crate::Result;
use ring::signature::{Ed25519KeyPair, KeyPair};
use std::{io::Read, path::Path};

pub fn publisher() -> Publisher {
    let key = Ed25519KeyPair::from_seed_unchecked(&[42; 32]).unwrap();
    Publisher {
        development: false,
        name: "Flare test publisher".into(),
        namespace: "flare-tests".into(),
        public_key: key.public_key().as_ref().try_into().unwrap(),
    }
}
pub fn authorize(path: impl AsRef<Path>) -> AuthorizedPackage {
    VerifiedPackage::open(path.as_ref(), &[publisher()])
        .unwrap()
        .authorize("flare-tests", false)
        .unwrap()
}
pub fn inspect(
    path: &Path,
    namespace: &str,
    verify_weights: bool,
    publishers: &[Publisher],
) -> Result<Vec<(u64, String)>> {
    let package = VerifiedPackage::open(path, publishers)?.authorize(namespace, verify_weights)?;
    let mut contents = Vec::new();
    let mut spans = Vec::new();
    for rank in &package.index.ranks {
        for name in rank
            .modules
            .iter()
            .map(|m| &m.file)
            .chain(rank.graphs.iter().map(|g| &g.resource))
            .chain(
                rank.modules
                    .iter()
                    .flat_map(|m| m.globals.iter().map(|g| &g.file)),
            )
        {
            spans.push(&rank.resources[name]);
        }
    }
    spans.extend(
        package
            .index
            .assets
            .iter()
            .filter_map(|asset| asset.payload.as_ref()),
    );
    for span in spans {
        let mut reader = package.archive.protection.reader(
            package.archive.file.clone(),
            package.archive.start,
            span,
        )?;
        let mut digest = ring::digest::Context::new(&ring::digest::SHA256);
        let mut buffer = vec![0; 65536];
        loop {
            let count = reader.read(&mut buffer).map_err(|e| e.to_string())?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
        contents.push((
            span.offset,
            super::protection::hex(digest.finish().as_ref()),
        ));
    }
    Ok(contents)
}

pub fn inspect_graph(path: &Path, name: &str) -> Result<()> {
    let mut package =
        VerifiedPackage::open(path, &[publisher()])?.authorize("flare-tests", false)?;
    let bundle = super::protobuf::decode(package.index.ranks.remove(0))?;
    let graph = bundle
        .graphs
        .iter()
        .find(|g| g.name == name)
        .ok_or("missing graph")?;
    let mut records = package.archive.records(&bundle, &graph.resource)?;
    for _ in 0..graph.node_count {
        records.read::<super::protobuf::wire::Node>()?;
    }
    for _ in 0..graph.edge_count {
        records.read::<super::protobuf::wire::Edge>()?;
    }
    records.finish()
}
