//! Cross-process CUDA graph replay with every allocation verified to relocate.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use flare_helios::{ReplayWorker, runtime_package::NativeResources};
use serde::Deserialize;

#[derive(Deserialize)]
struct Expected {
    addresses: HashMap<String, usize>,
    outputs: HashMap<String, String>,
    modified_outputs: HashMap<String, String>,
    pointers: Vec<Reference>,
}

#[derive(Deserialize)]
struct Reference {
    allocation: String,
    offset: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("expected fixture directory")?,
    );
    let expected: Expected = serde_json::from_slice(&std::fs::read(root.join("expected.json"))?)?;
    let owner = unsafe { NativeResources::load(root.join("package"), 0)? };
    for (name, original) in &expected.addresses {
        assert_ne!(
            owner.allocation(name)?.address,
            *original,
            "allocation {name} did not relocate"
        );
    }
    let pointers = owner.download("pointers")?;
    for (entry, reference) in pointers.chunks_exact(8).zip(&expected.pointers) {
        assert_eq!(
            u64::from_le_bytes(entry.try_into()?),
            (owner.allocation(&reference.allocation)?.address + reference.offset) as u64
        );
    }
    let driver = unsafe { libloading::Library::new("libcuda.so.1")? };
    let memory_info = unsafe {
        driver.get::<unsafe extern "C" fn(*mut usize, *mut usize) -> i32>(b"cuMemGetInfo_v2\0")?
    };
    let available = || {
        let (mut free, mut total) = (0, 0);
        assert_eq!(unsafe { memory_info(&mut free, &mut total) }, 0);
        free
    };
    let worker = unsafe { ReplayWorker::new(owner.graph_bundle(), 1)? };
    let poison = vec![0xff; owner.allocation("scratch")?.nbytes];
    let mut baseline = None;
    for iteration in 0..32 {
        if iteration == 16 {
            owner.upload("input", &std::fs::read(root.join("modified-input.bin"))?)?;
        }
        let outputs = if iteration < 16 {
            &expected.outputs
        } else {
            &expected.modified_outputs
        };
        for (index, name) in owner.graph_names().iter().enumerate() {
            owner.upload("scratch", &poison)?;
            let ticket = worker.submit(index, 3)?;
            let result = worker
                .poll(Duration::from_secs(30))?
                .ok_or("replay timed out")?;
            assert_eq!(result.ticket, ticket);
            if let Some(error) = result.error {
                return Err(error.into());
            }
            let reference = std::fs::read(root.join(&outputs[name]))?;
            let output = owner.download(name)?;
            assert_eq!(output, reference, "output differs for {name}");
        }
        if iteration == 0 {
            baseline = Some(available());
        }
    }
    assert_eq!(
        Some(available()),
        baseline,
        "GPU allocation changed during repeated replay"
    );
    let shutdown = worker.close();
    assert!(shutdown.drained, "native replay did not drain");
    if let Some(error) = shutdown.error {
        return Err(error.into());
    }
    let maps = std::fs::read_to_string("/proc/self/maps")?;
    assert!(
        !maps.contains("libpython")
            && !maps.contains("libtorch")
            && !maps.contains("site-packages")
    );
    drop(owner);
    println!(
        "native replay passed: relocated allocations, pointer tables, 64 alternating submissions, 192 graph launches, changed inputs, poisoned scratch, stable GPU memory; no Python or Torch mappings"
    );
    Ok(())
}
