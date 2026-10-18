use std::{
    net::{IpAddr, SocketAddr},
    num::NonZeroUsize,
    path::PathBuf,
};

use clap::{Args, Parser, Subcommand, builder::PossibleValuesParser, builder::TypedValueParser};
use flare_helios::{
    engram::Storage,
    offload::{PrefixCacheOffload, Transport, parse_size},
    runtime_package::{KvCacheLimit, LoadOptions, MemoryLimit},
};

#[derive(Debug, Parser)]
#[command(
    name = "flare",
    version,
    about = "Serve prepared models on NVIDIA GPUs"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Serve a local package or download a prepared package from Hugging Face.
    Run(Run),
    /// Download a prepared package from the hardware's Hugging Face revision.
    Download(Download),
    /// Accept rank zero's packages and executables and run its peer ranks on this host.
    Agent(AgentArgs),
    /// Manage remembered publisher trust.
    Trust {
        #[command(subcommand)]
        command: TrustCommand,
    },
}

#[derive(Debug, Args, Default)]
pub struct HubOptions {
    /// Hub revision; defaults to gb10 or gb202 detected from the selected GPUs.
    #[arg(long, value_name = "REVISION", help_heading = "Download options")]
    pub revision: Option<String>,

    /// Package filename; required when the revision contains several .safetensors files.
    #[arg(long, value_name = "FILE", help_heading = "Download options")]
    pub filename: Option<String>,

    /// Hugging Face hub cache [default: ~/.cache/huggingface/hub].
    #[arg(long, value_name = "PATH", help_heading = "Download options")]
    pub cache_dir: Option<PathBuf>,

    /// Hugging Face access token file; omit for public repositories.
    #[arg(long, value_name = "PATH", help_heading = "Download options")]
    pub token_file: Option<PathBuf>,

    /// Use only cached files, without contacting Hugging Face.
    #[arg(long, help_heading = "Download options")]
    pub offline: bool,
}

#[derive(Debug, Args)]
#[command(args_override_self = true)]
pub struct Download {
    /// Hugging Face repository containing prepared Flare packages, optionally ORG/NAME@REVISION.
    #[arg(value_name = "REPOSITORY")]
    pub repository: String,

    /// GPUs used to detect the hardware revision; unnecessary with --revision.
    #[arg(long, value_name = "N[,N...]", value_delimiter = ',', action = clap::ArgAction::Set,
        default_value = "0", value_parser = clap::value_parser!(i32).range(0..))]
    pub gpus: Vec<i32>,

    #[command(flatten)]
    pub hub: HubOptions,
}

#[derive(Debug, Args)]
pub struct AgentArgs {
    /// Address and port to listen on.
    #[arg(long, value_name = "IP:PORT", default_value = "0.0.0.0:29900")]
    pub listen: String,

    /// The cluster's shared token, readable only by its owner [default: ~/.config/flare/agent.token].
    #[arg(long, value_name = "PATH")]
    pub token_file: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
pub enum TrustCommand {
    /// Remove all remembered publisher identities for a signing key.
    Revoke {
        fingerprint: String,
        #[arg(long, value_name = "PATH")]
        trust_store: Option<PathBuf>,
    },
}

#[derive(Debug, Args)]
#[command(
    args_override_self = true,
    after_help = "Examples:\n  flare run model.safetensors --gpus 0 --host 127.0.0.1 --port 8000\n  flare run model.safetensors --embedding-table-storage ram --mtp-draft-tokens 3\n\nConcurrency and draft limits must fit the package's prepared capacities.\nOmitted concurrency and speculation options use package defaults."
)]
pub struct Run {
    /// Additional publisher verification profiles (JSON); does not grant execution trust.
    #[arg(long, value_name = "PATH", action = clap::ArgAction::Append)]
    pub publisher_keys: Vec<PathBuf>,
    /// Verify stored allocation and table ranges before loading model GPU resources.
    #[arg(long)]
    pub verify_weights: bool,
    /// Permanently accept this exact signing key together with --dangerously-accept-publisher.
    #[arg(
        long,
        requires = "dangerously_accept_publisher",
        value_name = "FINGERPRINT"
    )]
    pub dangerously_approve_key: Option<String>,
    /// Exact Hugging Face publisher namespace acknowledged for unattended first use.
    #[arg(long, requires = "dangerously_approve_key", value_name = "NAMESPACE")]
    pub dangerously_accept_publisher: Option<String>,
    /// Persistent trust file [default: ~/.config/flare/trust.toml].
    #[arg(long, value_name = "PATH")]
    pub trust_store: Option<PathBuf>,

    /// Local prepared .safetensors package or Hugging Face ORG/NAME[@REVISION].
    #[arg(value_name = "PACKAGE")]
    pub package: PathBuf,

    #[command(flatten)]
    pub hub: HubOptions,

    /// IP address for the HTTP server.
    #[arg(
        long,
        value_name = "ADDR",
        default_value = "127.0.0.1",
        help_heading = "Serving options"
    )]
    pub host: IpAddr,

    /// Port for the HTTP server.
    #[arg(
        long,
        value_name = "N",
        default_value = "8000",
        help_heading = "Serving options"
    )]
    pub port: u16,

    /// Replace the package's serving alias; the canonical model ID remains accepted.
    #[arg(long, value_name = "ALIAS", value_parser = served_model_name, help_heading = "Serving options")]
    pub served_model_name: Option<String>,

    /// GPUs in rank order, relative to CUDA_VISIBLE_DEVICES when set.
    #[arg(long, value_name = "N[,N...]", value_delimiter = ',', action = clap::ArgAction::Set,
        default_value = "0", value_parser = clap::value_parser!(i32).range(0..),
        help_heading = "Serving options")]
    pub gpus: Vec<i32>,

    /// Limit KV capacity by available memory, one context, or context times concurrency.
    #[arg(
        long,
        default_value = "memory",
        value_name = "MODE",
        help_heading = "Serving options",
        value_parser = PossibleValuesParser::new(["concurrency", "context", "memory"])
            .map(|value| match value.as_str() {
                "concurrency" => KvCacheLimit::Concurrency,
                "context" => KvCacheLimit::Context,
                _ => KvCacheLimit::Memory,
            })
    )]
    pub kv_cache_limit: KvCacheLimit,

    /// Per-GPU serving memory: occupied size (30G), free reserve (-30G), or total fraction (0.8).
    /// Omitted: package serving reserve.
    #[arg(
        long,
        allow_hyphen_values = true,
        value_name = "LIMIT",
        help_heading = "Serving options"
    )]
    pub memory_limit: Option<MemoryLimit>,

    /// Peer hosts of a multi-host package, ranks 1 onward, each running `flare agent`. Flare sends
    /// the package and itself to hosts that lack them, starts their ranks with these serving
    /// options, and serves as rank zero.
    #[arg(
        long,
        value_name = "HOST[,HOST...]",
        value_delimiter = ',',
        help_heading = "Multi-host options"
    )]
    pub hosts: Vec<String>,

    /// Experimental ring in rank order: only neighbor RoCE QPs connect;
    /// native CPU relays forward non-neighbor traffic. Also emulates a ring on a switch.
    #[arg(long, help_heading = "Multi-host options")]
    pub roce_ring: bool,

    /// The cluster's shared agent token [default: ~/.config/flare/agent.token].
    #[arg(long, value_name = "PATH", help_heading = "Multi-host options")]
    pub agent_token_file: Option<PathBuf>,

    /// Rank zero's rendezvous address: with --hosts, where rank zero listens [default:
    /// 0.0.0.0:29800]; on a peer, where it joins rank zero.
    #[arg(long, value_name = "HOST:PORT", help_heading = "Multi-host options")]
    pub rendezvous: Option<String>,

    /// This host's rank of a multi-host package; rank zero's launcher sets it for each peer.
    #[arg(long, value_name = "N", help_heading = "Multi-host options")]
    pub rank: Option<usize>,

    /// File containing the serving API key; omit to disable authentication.
    #[arg(long, value_name = "PATH", help_heading = "Serving options")]
    pub api_key_file: Option<PathBuf>,

    /// Maximum simultaneous requests [default: package setting].
    #[arg(long, value_name = "N", help_heading = "Serving options")]
    pub max_concurrency: Option<NonZeroUsize>,

    /// Favor token throughput or balance it with output cadence; both retain token-service fairness.
    #[arg(long, value_name = "MODE", default_value = "balanced",
        value_parser = ["balanced", "throughput"], help_heading = "Serving options")]
    pub scheduler_policy: String,

    /// Storage for auxiliary embedding tables (PLE/Engram) [default: disk].
    #[arg(long, value_name = "MODE", help_heading = "Serving options",
        value_parser = PossibleValuesParser::new(["disk", "ram"])
            .map(|value| if value == "disk" { Storage::Disk } else { Storage::Ram }))]
    pub embedding_table_storage: Option<Storage>,

    /// Per-host budget for a disk-backed embedding row cache, e.g. 1G; includes the
    /// index, usage counters and miss workspace. Selects disk storage; 0 disables [default: 0].
    #[arg(long, value_name = "SIZE", value_parser = parse_size, default_value = "0", help_heading = "Serving options")]
    pub embedding_table_cache_size: u64,

    /// Disk workers per embedding table; shares a fixed I/O budget [default: 4 on Windows, 1 elsewhere].
    #[arg(long, value_name = "N", help_heading = "Advanced serving options",
        value_parser = PossibleValuesParser::new(["1", "2", "4", "8"]).map(|value| value.parse::<usize>().unwrap()))]
    pub embedding_table_workers: Option<usize>,

    /// Maximum drafts per round; 0 disables speculation [default: package setting].
    #[arg(long, value_name = "N", help_heading = "Speculation options")]
    pub mtp_draft_tokens: Option<usize>,

    /// Adapt verification width to acceptance estimates and execution cost [default: package setting].
    #[arg(
        long,
        overrides_with = "no_mtp_adaptive_verification",
        help_heading = "Speculation options"
    )]
    pub mtp_adaptive_verification: bool,

    /// Use the maximum width where captured, otherwise width one [default: package setting].
    #[arg(
        long,
        overrides_with = "mtp_adaptive_verification",
        help_heading = "Speculation options"
    )]
    pub no_mtp_adaptive_verification: bool,

    /// Host memory for prefix checkpoints, e.g. 16G; 0 disables [default: disabled].
    #[arg(long, value_name = "SIZE", value_parser = parse_size, help_heading = "Prefix cache options")]
    pub prefix_cache_memory: Option<u64>,

    /// Directory for prefix checkpoints on disk [default: disabled].
    #[arg(long, value_name = "DIR", help_heading = "Prefix cache options")]
    pub prefix_cache_disk: Option<PathBuf>,

    /// Disk retention limit, e.g. 100G; 0 disables [default: 10% of free space].
    #[arg(long, value_name = "SIZE", value_parser = parse_size, help_heading = "Prefix cache options")]
    pub prefix_cache_disk_limit: Option<u64>,

    /// Transfer disk checkpoints through host staging or GPUDirect Storage [default: staged].
    #[arg(long, value_name = "MODE", help_heading = "Prefix cache options",
        value_parser = PossibleValuesParser::new(["staged", "direct"])
            .map(|value| if value == "staged" { Transport::Staged } else { Transport::Direct }))]
    pub prefix_cache_disk_transport: Option<Transport>,
}

fn served_model_name(value: &str) -> Result<String, String> {
    if value.trim().is_empty() {
        Err("served model name must not be empty".into())
    } else {
        Ok(value.to_owned())
    }
}

impl Run {
    pub fn bind_address(&self) -> SocketAddr {
        SocketAddr::new(self.host, self.port)
    }

    /// Serving options every rank of a multi-host package must load with: they select the same
    /// graphs and capacities on every host.
    pub fn rank_options(&self) -> Vec<String> {
        let mut options = Vec::new();
        if self.kv_cache_limit != KvCacheLimit::Memory {
            options.extend(["--kv-cache-limit".into(), self.kv_cache_limit.to_string()]);
        }
        if let Some(limit) = self.memory_limit {
            options.push(format!("--memory-limit={limit}"));
        }
        if self.roce_ring {
            options.push("--roce-ring".into());
        }
        if self.verify_weights {
            options.push("--verify-weights".into());
        }
        if let Some(n) = self.max_concurrency {
            options.extend(["--max-concurrency".into(), n.to_string()]);
        }
        if let Some(n) = self.mtp_draft_tokens {
            options.extend(["--mtp-draft-tokens".into(), n.to_string()]);
        }
        if self.mtp_adaptive_verification {
            options.push("--mtp-adaptive-verification".into());
        }
        if self.no_mtp_adaptive_verification {
            options.push("--no-mtp-adaptive-verification".into());
        }
        if let Some(storage) = self.embedding_table_storage {
            options.extend([
                "--embedding-table-storage".into(),
                if storage == Storage::Disk {
                    "disk"
                } else {
                    "ram"
                }
                .into(),
            ]);
        }
        if let Some(workers) = self.embedding_table_workers {
            options.extend(["--embedding-table-workers".into(), workers.to_string()]);
        }
        if self.embedding_table_cache_size != 0 {
            options.extend([
                "--embedding-table-cache-size".into(),
                self.embedding_table_cache_size.to_string(),
            ]);
        }
        options
    }

    pub fn load_options(&self) -> LoadOptions {
        LoadOptions {
            kv_cache_limit: self.kv_cache_limit,
            memory_limit: self.memory_limit,
            embedding_table_storage: self.embedding_table_storage,
            embedding_table_cache_bytes: self.embedding_table_cache_size,
            embedding_table_workers: self.embedding_table_workers,
            max_concurrency: self.max_concurrency.map(NonZeroUsize::get),
            mtp_draft_tokens: self.mtp_draft_tokens,
            adaptive_verification: if self.mtp_adaptive_verification {
                Some(true)
            } else if self.no_mtp_adaptive_verification {
                Some(false)
            } else {
                None
            },
            device_labels: Some(self.gpus.clone()),
            prefix_cache: PrefixCacheOffload {
                memory_bytes: self.prefix_cache_memory,
                disk: self.prefix_cache_disk.clone(),
                disk_limit: self.prefix_cache_disk_limit,
                transport: self.prefix_cache_disk_transport,
            },
            cluster_rank: self.rank,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(arguments: &[&str]) -> Run {
        let Command::Run(args) = Cli::try_parse_from(arguments).unwrap().command else {
            panic!("expected run")
        };
        args
    }

    #[test]
    fn embedding_workers_reach_loader_and_remote_arguments() {
        assert_eq!(
            run(&["flare", "run", "model.safetensors"])
                .load_options()
                .embedding_table_workers,
            None
        );
        for workers in ["1", "2", "4", "8"] {
            let args = run(&[
                "flare",
                "run",
                "model.safetensors",
                "--embedding-table-workers",
                workers,
            ]);
            assert_eq!(
                args.load_options().embedding_table_workers,
                Some(workers.parse().unwrap())
            );
            assert_eq!(args.load_options().embedding_workers(), workers.parse::<usize>().unwrap());
            assert!(
                args.rank_options()
                    .windows(2)
                    .any(|pair| pair == ["--embedding-table-workers", workers])
            );
        }
        for workers in ["0", "3", "5"] {
            assert!(
                Cli::try_parse_from([
                    "flare",
                    "run",
                    "model.safetensors",
                    "--embedding-table-workers",
                    workers
                ])
                .is_err()
            );
        }
    }

    #[test]
    fn omitted_limits_preserve_package_defaults() {
        let args = run(&["flare", "run", "model.safetensors"]);
        assert_eq!(args.bind_address().to_string(), "127.0.0.1:8000");
        assert_eq!(args.served_model_name, None);
        assert_eq!(args.scheduler_policy, "balanced");
        let options = args.load_options();
        assert_eq!(options.device_labels, Some(vec![0]));
        assert_eq!(options.embedding_table_storage, None);
        assert_eq!(options.embedding_table_cache_bytes, 0);
        assert_eq!(options.max_concurrency, None);
        assert_eq!(options.kv_cache_limit, KvCacheLimit::Memory);
        assert_eq!(options.memory_limit, None);
        assert_eq!(options.mtp_draft_tokens, None);
        assert_eq!(options.adaptive_verification, None);
        assert_eq!(options.prefix_cache, PrefixCacheOffload::default());
    }

    #[test]
    fn scheduling_preference_is_explicit_and_validated() {
        assert_eq!(run(&["flare", "run", "model.safetensors", "--scheduler-policy", "throughput"]).scheduler_policy,
                   "throughput");
        assert!(Cli::try_parse_from(["flare", "run", "model.safetensors", "--scheduler-policy", "unknown"]).is_err());
    }

    #[test]
    fn explicit_options_reach_the_loader() {
        let args = run(&[
            "flare",
            "run",
            "--gpus=1",
            "--gpus=2,0",
            "--host",
            "::1",
            "--port",
            "8001",
            "model.safetensors",
            "--api-key-file",
            "key.txt",
            "--max-concurrency",
            "3",
            "--mtp-draft-tokens=0",
            "--embedding-table-storage",
            "ram",
            "--prefix-cache-memory",
            "1.5GiB",
            "--prefix-cache-disk",
            "prefixes",
            "--prefix-cache-disk-limit",
            "0",
            "--prefix-cache-disk-transport",
            "direct",
        ]);
        assert_eq!(args.package, PathBuf::from("model.safetensors"));
        assert_eq!(args.bind_address().to_string(), "[::1]:8001");
        assert_eq!(args.api_key_file, Some(PathBuf::from("key.txt")));
        let options = args.load_options();
        assert_eq!(options.device_labels, Some(vec![2, 0]));
        assert_eq!(options.embedding_table_storage, Some(Storage::Ram));
        assert_eq!(options.max_concurrency, Some(3));
        assert_eq!(options.mtp_draft_tokens, Some(0));
        assert_eq!(
            options.prefix_cache,
            PrefixCacheOffload {
                memory_bytes: Some(3 << 29),
                disk: Some(PathBuf::from("prefixes")),
                disk_limit: Some(0),
                transport: Some(Transport::Direct),
            }
        );
    }

    #[test]
    fn ring_flag_is_forwarded_to_every_agent_started_rank() {
        let args = run(&["flare", "run", "model.safetensors", "--roce-ring"]);
        assert!(args.roce_ring);
        assert_eq!(args.rank_options(), vec!["--roce-ring"]);
    }

    #[test]
    fn last_verification_flag_wins() {
        let enable = "--mtp-adaptive-verification";
        let disable = "--no-mtp-adaptive-verification";
        for (flags, expected) in [
            (vec![enable], true),
            (vec![disable], false),
            (vec![enable, disable], false),
            (vec![disable, enable], true),
        ] {
            let mut arguments = vec!["flare", "run", "model.safetensors"];
            arguments.extend(flags);
            assert_eq!(
                run(&arguments).load_options().adaptive_verification,
                Some(expected)
            );
        }
    }

    #[test]
    fn listener_flags_support_independent_overrides_and_ipv6() {
        for (flags, expected) in [
            (vec!["--host", "0.0.0.0"], "0.0.0.0:8000"),
            (vec!["--port", "8001"], "127.0.0.1:8001"),
            (vec!["--host", "::1", "--port", "8002"], "[::1]:8002"),
        ] {
            let mut arguments = vec!["flare", "run", "model.safetensors"];
            arguments.extend(flags);
            assert_eq!(run(&arguments).bind_address().to_string(), expected);
        }
    }

    #[test]
    fn served_model_name_accepts_aliases_and_rejects_blank_values() {
        for name in ["my-model", "organization/model", "local model"] {
            let args = run(&[
                "flare",
                "run",
                "model.safetensors",
                "--served-model-name",
                name,
            ]);
            assert_eq!(args.served_model_name.as_deref(), Some(name));
        }
        for name in ["", " ", "\t"] {
            assert!(
                Cli::try_parse_from([
                    "flare",
                    "run",
                    "model.safetensors",
                    "--served-model-name",
                    name
                ])
                .is_err()
            );
        }
    }

    #[test]
    fn embedding_row_cache_accepts_a_binary_memory_budget() {
        let args = run(&[
            "flare",
            "run",
            "model.safetensors",
            "--embedding-table-cache-size",
            "1G",
        ]);
        let options = args.load_options();
        assert_eq!(options.embedding_table_cache_bytes, 1 << 30);
        assert_eq!(options.embedding_table_storage, None);
        assert_eq!(
            args.rank_options(),
            ["--embedding-table-cache-size", "1073741824"]
        );
    }

    #[test]
    fn memory_limits_accept_signed_values_and_reach_every_rank() {
        for value in ["30g", "-30g", "0.8"] {
            for flags in [
                vec!["--memory-limit".to_owned(), value.to_owned()],
                vec![format!("--memory-limit={value}")],
            ] {
                let mut arguments = vec![
                    "flare",
                    "run",
                    "model.safetensors",
                    "--kv-cache-limit",
                    "context",
                ];
                arguments.extend(flags.iter().map(String::as_str));
                let args = run(&arguments);
                assert_eq!(args.load_options().kv_cache_limit, KvCacheLimit::Context);
                assert_eq!(
                    args.load_options().memory_limit,
                    Some(value.parse().unwrap())
                );
                let forwarded = args.rank_options();
                let mut peer = vec!["flare", "run", "model.safetensors"];
                peer.extend(forwarded.iter().map(String::as_str));
                let peer = run(&peer).load_options();
                assert_eq!(peer.memory_limit, args.load_options().memory_limit);
                assert_eq!(peer.kv_cache_limit, KvCacheLimit::Context);
            }
        }
        for value in ["0", "-0.8", "2", "NaN", "1e-1", "-0G"] {
            assert!(
                Cli::try_parse_from(["flare", "run", "model.safetensors", "--memory-limit", value])
                    .is_err()
            );
        }
        assert!(
            Cli::try_parse_from([
                "flare",
                "run",
                "model.safetensors",
                "--kv-cache-limit",
                "unknown"
            ])
            .is_err()
        );
    }
}
