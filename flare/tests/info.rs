use std::{
    path::PathBuf,
    process::{Command, Output},
};

fn flare(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_flare"))
        .args(arguments)
        .output()
        .unwrap()
}

fn varint(mut n: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    while n >= 128 {
        bytes.push(n as u8 | 128);
        n >>= 7;
    }
    bytes.push(n as u8);
    bytes
}

fn integer(field: u64, n: u64) -> Vec<u8> {
    [varint(field << 3), varint(n)].concat()
}

fn message(field: u64, bytes: &[u8]) -> Vec<u8> {
    [
        varint(field << 3 | 2),
        varint(bytes.len() as u64),
        bytes.to_vec(),
    ]
    .concat()
}

fn fixture(directory: &std::path::Path, version: u64, maximum: u64) -> PathBuf {
    let allocation = [
        message(1, b"cache"),
        integer(3, 256),
        integer(7, 0x200000),
        message(
            9,
            &[integer(1, 1), integer(2, 300), integer(3, maximum)].concat(),
        ),
    ]
    .concat();
    let fixed = [
        message(1, b"weights"),
        integer(2, 4096),
        integer(3, 256),
        message(4, b"weights.bin"),
        message(5, &integer(2, 4096)),
    ]
    .concat();
    let payload = [
        message(1, b"weights.bin"),
        message(
            2,
            &[integer(2, 4096), message(3, b"absent-data.safetensors")].concat(),
        ),
    ]
    .concat();
    let graph = [message(1, b"decode-1"), integer(4, 12), integer(5, 11)].concat();
    let execution_graph = [
        message(1, b"decode-1"),
        message(2, b"decode"),
        message(3, b"sample"),
        integer(4, 1),
        integer(5, 1),
    ]
    .concat();
    let bundle = [
        message(1, &[12, 0]),
        message(2, &allocation),
        message(2, &fixed),
        message(5, &graph),
        message(7, &[integer(2, 4), message(5, &execution_graph)].concat()),
        message(9, &payload),
    ]
    .concat();
    let contract = [
        message(1, b"owner/model"),
        integer(4, 32768),
        integer(5, 4096),
        integer(16, 2),
        message(17, b"local-model"),
    ]
    .concat();
    // Optional explicit zeroes and absent fields must remain distinguishable.
    let sampling = [
        integer(1, u64::MAX),
        varint(2 << 3 | 5),
        0f32.to_le_bytes().to_vec(),
        integer(3, u64::MAX),
    ]
    .concat();
    let index = [
        integer(1, version),
        message(7, &sampling),
        message(10, &contract),
    ]
    .concat();
    let metadata = [
        message(1, &index),
        message(2, b"example"),
        message(3, b"https://huggingface.co/example/model"),
        integer(6, 2),
    ]
    .concat();
    let group = |kind, body: &[u8]| message(4, &[integer(1, kind), message(2, body)].concat());
    let rank = |n| {
        message(
            5,
            &[
                integer(1, 4),
                message(2, &[integer(1, n), message(2, &bundle)].concat()),
            ]
            .concat(),
        )
    };
    let envelope = [
        integer(1, 4),
        group(1, &metadata),
        group(2, &[]),
        group(3, &[]),
        rank(0),
        rank(1),
    ]
    .concat();
    let header = serde_json::json!({
        "__metadata__": {"format": "flare"},
        "flare.payload": {"dtype": "U8", "shape": [0], "data_offsets": [0, 0]},
        "flare.metadata": {"dtype": "U8", "shape": [envelope.len()], "data_offsets": [0, envelope.len()]},
    });
    let mut bytes = 4088u64.to_le_bytes().to_vec();
    bytes.extend(serde_json::to_vec(&header).unwrap());
    bytes.resize(4096, b' ');
    bytes.extend(envelope);
    let path = directory.join(format!("model-{version}.safetensors"));
    std::fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn inspection_reads_primary_metadata_without_cuda_trust_or_companions() {
    let temp = tempfile::tempdir().unwrap();
    for version in [4, 5] {
        let path = fixture(temp.path(), version, 10);
        let path = path.to_str().unwrap();
        let basic = flare(&["info", path]);
        assert!(basic.status.success(), "{basic:?}");
        assert!(basic.stderr.is_empty(), "{basic:?}");
        let text = String::from_utf8(basic.stdout).unwrap();
        for expected in [
            "owner/model",
            "local-model",
            "32768",
            "Sampling defaults",
            "unverified",
            "Rank 0",
            "Rank 1",
        ] {
            assert!(text.contains(expected), "{expected}: {text}");
        }
        assert!(!text.contains("Captured graphs"));
        assert!(!text.contains("Memory map"));
        for (flag, expected, omitted) in [
            ("--memory-map", "Memory map", "Captured graphs"),
            ("--graphs", "Captured graphs", "Memory map"),
            ("--all", "Frontend settings", "never-present"),
        ] {
            let output = flare(&["info", path, flag]);
            assert!(output.status.success(), "{output:?}");
            let text = String::from_utf8(output.stdout).unwrap();
            assert!(text.contains(expected), "{text}");
            assert!(!text.contains(omitted), "{text}");
            if flag == "--memory-map" {
                assert!(text.contains("0x0000000000200000"), "{text}");
                assert!(text.contains("3072"), "{text}");
                assert!(text.contains("absent-data.safetensors"), "{text}");
            }
        }
        let output = flare(&["info", path, "--json"]);
        assert!(output.status.success(), "{output:?}");
        let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(json["signatures_verified"], false);
        assert_eq!(json["sampling_defaults"]["temperature"], 0.0);
        assert_eq!(json["sampling_defaults"]["seed"], u64::MAX);
        assert_eq!(json["sampling_defaults"]["top_k"], -1);
        assert!(json["sampling_defaults"]["top_p"].is_null());
        assert_eq!(json["ranks"][0]["allocations"][0]["virtual_bytes"], 3072);
        assert_eq!(json["ranks"][1]["graphs"][0]["packed_tokens"], 1);
    }
}

#[test]
fn invalid_packages_fail_before_loading() {
    let temp = tempfile::tempdir().unwrap();
    let path = fixture(temp.path(), 5, u64::MAX);
    let output = flare(&["info", path.to_str().unwrap()]);
    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("allocation extent overflow"), "{error}");
    std::fs::write(&path, 65537u64.to_le_bytes()).unwrap();
    let output = flare(&["info", path.to_str().unwrap()]);
    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("invalid safetensors header size"), "{error}");
    let output = flare(&["info", "/nonexistent/package.safetensors"]);
    assert!(!output.status.success());
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .contains("cannot inspect package")
    );
}
