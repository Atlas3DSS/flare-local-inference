use blake2::digest::consts::U16;
use blake2::{Blake2b, Digest};

use crate::Result;

pub const PAGE_SIZE: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PrefixKey {
    pub consumed_tokens: u64,
    pub digest: [u8; 16],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConditioningEvent {
    pub position: u64,
    pub identity: [u8; 32],
}

/// Incremental token and conditioning identity for a single model/recipe.
///
/// Events belong to token positions in the appended segment. Events at the same
/// position retain caller order. A boundary at N contains positions less than N;
/// appending conditioning at N cannot change that boundary's digest.
#[derive(Clone, Debug)]
pub struct PrefixHistory {
    identity: [u8; 32],
    boundaries: Vec<PrefixKey>,
    tail: Vec<u32>,
    tail_events: Vec<ConditioningEvent>,
}

impl PrefixHistory {
    pub fn new(identity: [u8; 32]) -> Self {
        let mut hash = Blake2b::<U16>::new();
        hash.update(b"sparkgap-prefix-root-v1\0");
        hash.update(identity);
        Self {
            identity,
            boundaries: vec![PrefixKey {
                consumed_tokens: 0,
                digest: hash.finalize().into(),
            }],
            tail: Vec::with_capacity(PAGE_SIZE),
            tail_events: Vec::new(),
        }
    }

    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }

    pub fn token_count(&self) -> u64 {
        self.boundaries.last().unwrap().consumed_tokens + self.tail.len() as u64
    }

    pub fn key_at(&self, consumed_tokens: u64) -> Option<PrefixKey> {
        if !consumed_tokens.is_multiple_of(PAGE_SIZE as u64) {
            return None;
        }
        let index = usize::try_from(consumed_tokens / PAGE_SIZE as u64).ok()?;
        self.boundaries.get(index).copied()
    }

    pub fn aligned_boundaries(&self) -> &[PrefixKey] {
        &self.boundaries[1..]
    }

    pub fn append(&mut self, tokens: &[u32], events: &[ConditioningEvent]) -> Result<()> {
        let start = self.token_count();
        let end = start
            .checked_add(tokens.len() as u64)
            .ok_or("token length overflow")?;
        let mut last_position = start;
        for event in events {
            if event.position < last_position || event.position >= end {
                return Err(
                    "conditioning events must be ordered within the appended token segment".into(),
                );
            }
            last_position = event.position;
        }
        let mut event_index = 0;
        for (index, token) in tokens.iter().enumerate() {
            let position = start + index as u64;
            while event_index < events.len() && events[event_index].position == position {
                self.tail_events.push(events[event_index]);
                event_index += 1;
            }
            self.tail.push(*token);
            if self.tail.len() == PAGE_SIZE {
                let parent = *self.boundaries.last().unwrap();
                let consumed_tokens = parent.consumed_tokens + PAGE_SIZE as u64;
                let mut hash = Blake2b::<U16>::new();
                hash.update(b"sparkgap-prefix-block-v1\0");
                hash.update(self.identity);
                hash.update(parent.digest);
                hash.update(consumed_tokens.to_le_bytes());
                for token in &self.tail {
                    hash.update(token.to_le_bytes());
                }
                hash.update((self.tail_events.len() as u64).to_le_bytes());
                for event in &self.tail_events {
                    hash.update(event.position.to_le_bytes());
                    hash.update(event.identity);
                }
                self.boundaries.push(PrefixKey {
                    consumed_tokens,
                    digest: hash.finalize().into(),
                });
                self.tail.clear();
                self.tail_events.clear();
            }
        }
        Ok(())
    }
}
