use super::*;
use crate::ReplayWorker;
use std::time::Duration;

#[test]
#[ignore = "requires FLARE_COOPERATIVE_TEST_PACKAGE exported by cooperative_package_fixture.py on the assigned GPU"]
fn oversized_cooperative_grids_replay_at_device_capacity() {
    let root =
        std::path::PathBuf::from(std::env::var_os("FLARE_COOPERATIVE_TEST_PACKAGE").unwrap());
    let owner =
        unsafe { NativeResources::load(test_support::authorize(root.join("package")), 0).unwrap() };
    let expected: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("expected.json")).unwrap()).unwrap();
    let length = expected["length"].as_u64().unwrap() as usize;
    let worker = unsafe { ReplayWorker::new(owner.graph_bundle(), 1).unwrap() };
    let poison = vec![0xff; (length * 2 + 3) * 4];
    for _ in 0..4 {
        for (index, name) in owner.graph_names().iter().enumerate() {
            owner.upload(name, &poison).unwrap();
            worker.submit(index, 3).unwrap();
            let result = worker.poll(Duration::from_secs(10)).unwrap().unwrap();
            assert!(result.error.is_none(), "{:?}", result.error);
            let bytes = owner.download(name).unwrap();
            let values: Vec<i32> = bytes
                .chunks_exact(4)
                .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            for i in 0..length {
                assert_eq!(values[i], i as i32 * 3 + 7, "{name}: element {i}");
                assert_eq!(
                    values[length + i],
                    if name == "ordinary" {
                        -1
                    } else {
                        (length - 1 - i) as i32 * 3 + 18
                    },
                    "{name}: post-barrier element {i}",
                );
            }
            let grid: Vec<i32> = serde_json::from_value(expected["grids"][name].clone()).unwrap();
            assert_eq!(values[length * 2..], grid, "{name}: replayed grid");
        }
    }
    assert!(worker.close().drained);
}
