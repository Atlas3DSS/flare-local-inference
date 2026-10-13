//! Publisher identities provisioned by the runtime distributor.

use super::{Publisher, format::invalid};
use crate::Result;

const SLOT_SIZE: usize = 65536;
const MAGIC: &[u8] = b"FLARE-PUBLISHERS-V3";
const fn empty_slot() -> [u8; SLOT_SIZE] {
    let mut result = [0; SLOT_SIZE];
    let mut i = 0;
    while i < MAGIC.len() {
        result[i] = MAGIC[i];
        i += 1;
    }
    result
}
#[used]
#[unsafe(link_section = ".flare_publishers")]
static EMBEDDED: [u8; SLOT_SIZE] = empty_slot();

pub(super) fn embedded() -> Result<Vec<Publisher>> {
    // The release stamper fills this ELF section after linking. Volatile reads prevent
    // the compiler from replacing its contents with the source-build placeholders.
    let pointer = std::ptr::addr_of!(EMBEDDED).cast::<u8>();
    let slot: Vec<u8> = (0..SLOT_SIZE)
        .map(|i| unsafe { std::ptr::read_volatile(pointer.add(i)) })
        .collect();
    if &slot[..MAGIC.len()] != MAGIC {
        return Err(invalid("invalid embedded publisher section"));
    }
    let length = u32::from_le_bytes(slot[32..36].try_into().unwrap()) as usize;
    if length > SLOT_SIZE - 36 {
        return Err(invalid("embedded publisher section exceeds capacity"));
    }
    let publishers: Vec<Publisher> = if length == 0 {
        Vec::new()
    } else {
        serde_json::from_slice(&slot[36..36 + length]).map_err(|e| invalid(e.to_string()))?
    };
    for publisher in &publishers {
        publisher.validate()?;
    }
    Ok(publishers)
}
