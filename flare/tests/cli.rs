use std::process::{Command, Output};

fn flare(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_flare"))
        .args(arguments)
        .output()
        .unwrap()
}

#[test]
fn help_exits_successfully_before_accessing_files() {
    for arguments in [
        vec!["--help"],
        vec!["-h"],
        vec!["run", "--help"],
        vec!["run", "-h"],
        vec!["help", "run"],
        vec![
            "run",
            "/nonexistent/model.safetensors",
            "--embedding-table-storage",
            "disk",
            "--bind",
            "127.0.0.1:8000",
            "--api-key-file",
            "/nonexistent/key.txt",
            "--help",
        ],
    ] {
        let output = flare(&arguments);
        assert!(output.status.success(), "{arguments:?}: {output:?}");
        assert!(output.stderr.is_empty(), "{arguments:?}: {output:?}");
        let help = String::from_utf8(output.stdout).unwrap();
        assert!(help.contains("Usage: flare"), "{arguments:?}: {help}");
        if arguments.contains(&"run") {
            assert!(help.contains("--embedding-table-storage"), "{help}");
            assert!(help.contains("--no-mtp-adaptive-verification"), "{help}");
            assert!(help.contains("--prefix-cache-disk-transport"), "{help}");
            assert!(help.contains("127.0.0.1:8000"), "{help}");
        }
    }
}

#[test]
fn usage_errors_identify_the_problem_before_accessing_files() {
    for (arguments, expected) in [
        (vec![], "Usage: flare"),
        (vec!["run"], "<PACKAGE>"),
        (vec!["rnu"], "unrecognized subcommand"),
        (
            vec!["run", "/nonexistent/model", "--bogus"],
            "unexpected argument '--bogus'",
        ),
        (
            vec!["run", "/nonexistent/model", "--bind"],
            "a value is required for '--bind",
        ),
        (
            vec!["run", "/nonexistent/model", "--bind", "bad"],
            "invalid value 'bad' for '--bind",
        ),
        (
            vec![
                "run",
                "/nonexistent/model",
                "--embedding-table-storage",
                "bad",
            ],
            "possible values: disk, ram",
        ),
        (
            vec!["run", "/nonexistent/model", "--max-concurrency", "0"],
            "invalid value '0' for '--max-concurrency",
        ),
        (
            vec!["run", "/nonexistent/model", "--mtp-draft-tokens", "bad"],
            "invalid value 'bad' for '--mtp-draft-tokens",
        ),
        (
            vec!["run", "/nonexistent/model", "--prefix-cache-memory", "bad"],
            "invalid value 'bad' for '--prefix-cache-memory",
        ),
    ] {
        let output = flare(&arguments);
        assert_eq!(output.status.code(), Some(2), "{arguments:?}: {output:?}");
        assert!(output.stdout.is_empty(), "{arguments:?}: {output:?}");
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(error.contains(expected), "{arguments:?}: {error}");
        assert!(error.contains("--help"), "{arguments:?}: {error}");
    }
}
