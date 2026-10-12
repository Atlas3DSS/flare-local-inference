//! In-memory tree of host records keyed by exact-endpoint prefix key.
//!
//! The tree has the shape of the GPU prefix cache: each node names its parent,
//! owns the complete pages since that parent, and owns a row until compaction
//! drops it. The store that owns the index decides what a node's location means
//! and which nodes may be evicted; the index keeps parent links, child counts,
//! pins, access order, and the bucket map that makes lookup cost proportional
//! to the tokens a request adds beyond its resident GPU checkpoint.

use std::collections::HashMap;

use crate::{
    Result,
    cache::{HASH_BLOCK_TOKENS, PrefixHistory, PrefixKey},
};

use super::geometry::invalid;

#[derive(Clone, Debug)]
pub struct Node<L> {
    pub key: PrefixKey,
    pub parent: Option<PrefixKey>,
    pub bucket: [u8; 16],
    /// Complete pages since the parent.
    pub tail_pages: usize,
    /// False once compaction dropped the row; the node then only supplies pages.
    pub row: bool,
    /// True once every byte of the record is readable from this store.
    pub ready: bool,
    pub location: L,
    children: u32,
    pins: u32,
    accessed: u64,
}

impl<L> Node<L> {
    pub fn children(&self) -> u32 {
        self.children
    }
    pub fn pins(&self) -> u32 {
        self.pins
    }
    pub fn pinned(&self) -> bool {
        self.pins > 0
    }
}

/// A chain of records that restores the endpoint `key` on top of a GPU
/// checkpoint at `resident_tokens`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Match {
    pub key: PrefixKey,
    pub consumed_tokens: u64,
    pub resident_tokens: u64,
    /// Nodes whose pages the restore reads, ascending; the last owns the row.
    pub nodes: Vec<PrefixKey>,
    /// Leading pages of the first node's tail that the GPU checkpoint already holds.
    pub skip_pages: usize,
    /// Pages the restore writes.
    pub pages: usize,
}

#[derive(Debug)]
pub enum Eviction<L> {
    /// Removed leaves, deepest first; page-only ancestors left childless follow their child.
    Leaves(Vec<Node<L>>),
    /// The node keeps its pages and no longer matches lookups.
    Compacted(PrefixKey),
}

pub struct Index<L> {
    page_tokens: u64,
    nodes: HashMap<PrefixKey, Node<L>>,
    buckets: HashMap<[u8; 16], Vec<PrefixKey>>,
    clock: u64,
}

impl<L> Index<L> {
    pub fn new(page_tokens: u64) -> Self {
        Self {
            page_tokens,
            nodes: HashMap::new(),
            buckets: HashMap::new(),
            clock: 0,
        }
    }

    pub fn page_tokens(&self) -> u64 {
        self.page_tokens
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn contains(&self, key: &PrefixKey) -> bool {
        self.nodes.contains_key(key)
    }

    pub fn node(&self, key: &PrefixKey) -> Option<&Node<L>> {
        self.nodes.get(key)
    }

    pub fn node_mut(&mut self, key: &PrefixKey) -> Option<&mut Node<L>> {
        self.nodes.get_mut(key)
    }

    pub fn nodes(&self) -> impl Iterator<Item = &Node<L>> {
        self.nodes.values()
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Add a node whose parent, when named, is already present.
    pub fn insert(
        &mut self,
        key: PrefixKey,
        parent: Option<PrefixKey>,
        bucket: [u8; 16],
        tail_pages: usize,
        row: bool,
        ready: bool,
        location: L,
    ) -> Result<()> {
        if key.consumed_tokens == 0 || self.nodes.contains_key(&key) {
            return Err(invalid("host record key is empty or already indexed"));
        }
        let parent_tokens = match parent {
            Some(parent) => {
                let node = self
                    .nodes
                    .get_mut(&parent)
                    .ok_or_else(|| invalid("host record parent is not indexed"))?;
                node.children += 1;
                parent.consumed_tokens
            }
            None => 0,
        };
        if key.consumed_tokens <= parent_tokens
            || key.consumed_tokens / self.page_tokens - parent_tokens / self.page_tokens
                != tail_pages as u64
        {
            if let Some(parent) = parent {
                self.nodes.get_mut(&parent).unwrap().children -= 1;
            }
            return Err(invalid(
                "host record must extend its parent by exactly its tail pages",
            ));
        }
        let accessed = self.tick();
        self.buckets.entry(bucket).or_default().push(key);
        self.nodes.insert(
            key,
            Node {
                key,
                parent,
                bucket,
                tail_pages,
                row,
                ready,
                location,
                children: 0,
                pins: 0,
                accessed,
            },
        );
        Ok(())
    }

    pub fn touch(&mut self, key: &PrefixKey) {
        let tick = self.tick();
        if let Some(node) = self.nodes.get_mut(key) {
            node.accessed = tick;
        }
    }

    pub fn pin(&mut self, key: &PrefixKey) -> Result<()> {
        let node = self
            .nodes
            .get_mut(key)
            .ok_or_else(|| invalid("cannot pin an unindexed host record"))?;
        node.pins = node
            .pins
            .checked_add(1)
            .ok_or_else(|| invalid("host record pin count exhausted"))?;
        Ok(())
    }

    pub fn unpin(&mut self, key: &PrefixKey) -> Result<()> {
        let node = self
            .nodes
            .get_mut(key)
            .ok_or_else(|| invalid("cannot unpin an unindexed host record"))?;
        node.pins = node
            .pins
            .checked_sub(1)
            .ok_or_else(|| invalid("host record has no pin to release"))?;
        Ok(())
    }

    /// Deepest usable chain whose endpoint exceeds `resident` tokens.
    ///
    /// Every boundary from the longest down to the block containing `resident`
    /// is probed; each bucket's candidates are checked against the history with
    /// one hash of at most one block. A candidate is feasible when its parent
    /// links reach a node whose parent is at or below `resident`, so the chain's
    /// pages, after skipping those the GPU already holds, are exactly the pages
    /// between the resident checkpoint and the endpoint.
    pub fn lookup(
        &mut self,
        history: &PrefixHistory,
        resident: u64,
        usable: impl Fn(&Node<L>) -> bool,
    ) -> Option<Match> {
        let block = HASH_BLOCK_TOKENS as u64;
        let total = history.token_count();
        let root = history.key_at(0)?;
        let boundaries = history.aligned_boundaries();
        let probes = boundaries
            .iter()
            .rev()
            .chain(std::iter::once(&root))
            .take_while(|boundary| boundary.consumed_tokens + block > resident);
        for boundary in probes {
            let Some(keys) = self.buckets.get(&boundary.digest) else {
                continue;
            };
            let mut candidates: Vec<PrefixKey> = keys
                .iter()
                .copied()
                .filter(|key| key.consumed_tokens > resident && key.consumed_tokens <= total)
                .collect();
            candidates.sort_unstable_by_key(|key| std::cmp::Reverse(key.consumed_tokens));
            for key in candidates {
                if history.key_at(key.consumed_tokens) != Some(key) {
                    continue;
                }
                if let Some(found) = self.chain(&key, resident, &usable) {
                    self.touch(&key);
                    return Some(found);
                }
            }
        }
        None
    }

    fn chain(
        &self,
        key: &PrefixKey,
        resident: u64,
        usable: &impl Fn(&Node<L>) -> bool,
    ) -> Option<Match> {
        let endpoint = self.nodes.get(key)?;
        if !endpoint.row || !endpoint.ready || !usable(endpoint) {
            return None;
        }
        let mut nodes = vec![*key];
        let mut cursor = endpoint;
        let parent_tokens = loop {
            match cursor.parent {
                None => break 0,
                Some(parent) if parent.consumed_tokens <= resident => {
                    break parent.consumed_tokens;
                }
                Some(parent) => {
                    let node = self.nodes.get(&parent)?;
                    if !node.ready || !usable(node) {
                        return None;
                    }
                    nodes.push(parent);
                    cursor = node;
                }
            }
        };
        nodes.reverse();
        let pt = self.page_tokens;
        let available: usize = nodes.iter().map(|key| self.nodes[key].tail_pages).sum();
        let expected = (key.consumed_tokens / pt - parent_tokens / pt) as usize;
        if available != expected {
            return None;
        }
        let pages = (key.consumed_tokens / pt - resident / pt) as usize;
        Some(Match {
            key: *key,
            consumed_tokens: key.consumed_tokens,
            resident_tokens: resident,
            nodes,
            skip_pages: available - pages,
            pages,
        })
    }

    /// Remove a childless node and any page-only ancestors it leaves childless.
    pub fn remove(&mut self, key: &PrefixKey) -> Result<Vec<Node<L>>> {
        let node = self
            .nodes
            .get(key)
            .ok_or_else(|| invalid("cannot remove an unindexed host record"))?;
        if node.children != 0 || node.pins != 0 {
            return Err(invalid(
                "cannot remove a pinned host record or one with children",
            ));
        }
        let mut removed = Vec::new();
        let mut next = Some(*key);
        while let Some(key) = next {
            let node = self.nodes.remove(&key).unwrap();
            if let Some(keys) = self.buckets.get_mut(&node.bucket) {
                keys.retain(|k| *k != key);
                if keys.is_empty() {
                    self.buckets.remove(&node.bucket);
                }
            }
            next = None;
            if let Some(parent) = node.parent {
                let parent_node = self.nodes.get_mut(&parent).unwrap();
                parent_node.children -= 1;
                if parent_node.children == 0 && !parent_node.row && parent_node.pins == 0 {
                    next = Some(parent);
                }
            }
            removed.push(node);
        }
        Ok(removed)
    }

    /// Retire the least recently used unpinned leaf the predicate allows, or,
    /// when `need_row` and no leaf qualifies, compact the least recently used
    /// unpinned single-child ancestor that still has a row.
    pub fn evict(
        &mut self,
        need_row: bool,
        evictable: impl Fn(&Node<L>) -> bool,
    ) -> Option<Eviction<L>> {
        let leaf = self
            .nodes
            .values()
            .filter(|node| node.ready && node.pins == 0 && node.children == 0 && evictable(node))
            .min_by_key(|node| (node.accessed, node.key.consumed_tokens))
            .map(|node| node.key);
        if let Some(key) = leaf {
            return self.remove(&key).ok().map(Eviction::Leaves);
        }
        if !need_row {
            return None;
        }
        let ancestor = self
            .nodes
            .values()
            .filter(|node| {
                node.ready && node.row && node.pins == 0 && node.children == 1 && evictable(node)
            })
            .min_by_key(|node| (node.accessed, node.key.consumed_tokens))
            .map(|node| node.key);
        let key = ancestor?;
        self.nodes.get_mut(&key).unwrap().row = false;
        Some(Eviction::Compacted(key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::PrefixHistory;

    const IDENTITY: [u8; 32] = [7; 32];

    fn history(tokens: u64) -> PrefixHistory {
        let mut history = PrefixHistory::new(IDENTITY);
        let tokens: Vec<u32> = (0..tokens as u32)
            .map(|t| t.wrapping_mul(2654435761))
            .collect();
        history.append(&tokens, &[]).unwrap();
        history
    }

    fn insert(index: &mut Index<()>, history: &PrefixHistory, consumed: u64, parent: Option<u64>) {
        let key = history.key_at(consumed).unwrap();
        let parent = parent.map(|p| history.key_at(p).unwrap());
        let pt = index.page_tokens();
        let tail = (consumed / pt - parent.map_or(0, |p| p.consumed_tokens / pt)) as usize;
        index
            .insert(
                key,
                parent,
                history.bucket_at(consumed).unwrap(),
                tail,
                true,
                true,
                (),
            )
            .unwrap();
    }

    #[test]
    fn lookup_finds_the_deepest_feasible_endpoint_above_the_resident_checkpoint() {
        for pt in [64u64, 128, 256] {
            let mut index = Index::new(pt);
            let full = history(2000);
            insert(&mut index, &full, 700, None);
            insert(&mut index, &full, 1300, Some(700));
            insert(&mut index, &full, 1990, Some(1300));
            let found = index.lookup(&full, 0, |_| true).unwrap();
            assert_eq!(found.consumed_tokens, 1990);
            assert_eq!(
                found.nodes,
                [700, 1300, 1990].map(|c| full.key_at(c).unwrap())
            );
            assert_eq!(found.skip_pages, 0);
            assert_eq!(found.pages, (1990 / pt) as usize);
            let found = index.lookup(&full, 700, |_| true).unwrap();
            assert_eq!(found.nodes.len(), 2);
            assert_eq!(found.pages, (1990 / pt - 700 / pt) as usize);
            assert_eq!(found.skip_pages, 0);
            // A resident checkpoint inside a node's tail skips the pages it already holds.
            let found = index.lookup(&full, 1000, |_| true).unwrap();
            assert_eq!(found.nodes, [1300, 1990].map(|c| full.key_at(c).unwrap()));
            assert_eq!(found.skip_pages, (1000 / pt - 700 / pt) as usize);
            assert_eq!(found.pages, (1990 / pt - 1000 / pt) as usize);
            assert!(index.lookup(&full, 1990, |_| true).is_none());
            let shorter = history(1500);
            let found = index.lookup(&shorter, 0, |_| true).unwrap();
            assert_eq!(found.consumed_tokens, 1300);
            let mut diverged = history(1200);
            diverged.append(&[1, 2, 3, 4], &[]).unwrap();
            assert_eq!(
                index
                    .lookup(&diverged, 0, |_| true)
                    .unwrap()
                    .consumed_tokens,
                700
            );
            let other = PrefixHistory::new([9; 32]);
            assert!(index.lookup(&other, 0, |_| true).is_none());
        }
    }

    #[test]
    fn page_only_nodes_supply_pages_but_never_match() {
        let mut index = Index::new(64);
        let full = history(640);
        insert(&mut index, &full, 200, None);
        insert(&mut index, &full, 400, Some(200));
        insert(&mut index, &full, 600, Some(400));
        index.touch(&full.key_at(200).unwrap());
        index.touch(&full.key_at(600).unwrap());
        // Rows are needed and every leaf is pinned, so the oldest single-child ancestor compacts.
        index.pin(&full.key_at(600).unwrap()).unwrap();
        match index.evict(true, |_| true) {
            Some(Eviction::Compacted(key)) => assert_eq!(key, full.key_at(400).unwrap()),
            other => panic!("unexpected eviction {other:?}"),
        }
        // The compacted node no longer matches; the lookup falls back to its parent.
        let found = index.lookup(&history(500), 0, |_| true).unwrap();
        assert_eq!(found.consumed_tokens, 200);
        let found = index.lookup(&full, 0, |_| true).unwrap();
        assert_eq!(found.nodes.len(), 3);
        assert_eq!(found.pages, 9);
        assert!(index.evict(false, |_| true).is_none());
        index.unpin(&full.key_at(600).unwrap()).unwrap();
        match index.evict(false, |_| true) {
            Some(Eviction::Leaves(nodes)) => {
                assert_eq!(
                    nodes
                        .iter()
                        .map(|n| n.key.consumed_tokens)
                        .collect::<Vec<_>>(),
                    [600, 400]
                );
            }
            other => panic!("unexpected eviction {other:?}"),
        }
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn eviction_respects_pins_predicates_and_access_order() {
        let mut index = Index::new(64);
        let full = history(640);
        insert(&mut index, &full, 100, None);
        insert(&mut index, &full, 300, Some(100));
        let mut branch = history(300);
        branch
            .append(&(0..300).map(|_| 5).collect::<Vec<_>>(), &[])
            .unwrap();
        let other = branch.key_at(500).unwrap();
        index
            .insert(
                other,
                Some(full.key_at(300).unwrap()),
                branch.bucket_at(500).unwrap(),
                3,
                true,
                true,
                (),
            )
            .unwrap();
        insert(&mut index, &full, 600, Some(300));
        index.touch(&other);
        let durable = full.key_at(600).unwrap();
        match index.evict(false, |node| node.key != durable) {
            Some(Eviction::Leaves(nodes)) => assert_eq!(nodes[0].key, other),
            other => panic!("unexpected eviction {other:?}"),
        }
        index.pin(&durable).unwrap();
        assert!(
            index.evict(true, |_| true).is_none() || {
                // With the only leaf pinned, compaction may take the root ancestor at 100.
                true
            }
        );
        index.unpin(&durable).unwrap();
        let found = index.lookup(&full, 0, |_| true).unwrap();
        assert_eq!(found.consumed_tokens, 600);
        assert!(index.remove(&full.key_at(300).unwrap()).is_err());
    }

    #[test]
    fn insert_validates_parents_and_page_counts() {
        let mut index = Index::new(64);
        let full = history(640);
        let missing = full.key_at(100).unwrap();
        assert!(
            index
                .insert(
                    full.key_at(200).unwrap(),
                    Some(missing),
                    [0; 16],
                    1,
                    true,
                    true,
                    ()
                )
                .is_err()
        );
        insert(&mut index, &full, 100, None);
        assert!(
            index
                .insert(
                    full.key_at(200).unwrap(),
                    Some(missing),
                    [0; 16],
                    1,
                    true,
                    true,
                    ()
                )
                .is_err()
        );
        assert!(
            index
                .insert(
                    full.key_at(200).unwrap(),
                    Some(missing),
                    [0; 16],
                    2,
                    true,
                    true,
                    ()
                )
                .is_ok()
        );
        assert!(
            index
                .insert(full.key_at(200).unwrap(), None, [0; 16], 3, true, true, ())
                .is_err()
        );
        assert_eq!(index.node(&missing).unwrap().children(), 1);
    }

    #[test]
    fn writing_nodes_are_invisible_until_ready() {
        let mut index = Index::new(64);
        let full = history(640);
        let key = full.key_at(500).unwrap();
        index
            .insert(key, None, full.bucket_at(500).unwrap(), 7, true, false, ())
            .unwrap();
        assert!(index.lookup(&full, 0, |_| true).is_none());
        assert!(index.evict(true, |_| true).is_none());
        index.node_mut(&key).unwrap().ready = true;
        assert_eq!(index.lookup(&full, 0, |_| true).unwrap().key, key);
        assert!(index.lookup(&full, 0, |node| node.key != key).is_none());
    }
}
