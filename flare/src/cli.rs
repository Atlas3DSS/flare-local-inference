use std::{net::SocketAddr, num::NonZeroUsize, path::PathBuf};

use clap::{Args, Parser, Subcommand, builder::PossibleValuesParser, builder::TypedValueParser};
use flare_helios::{
    engram::Storage,
    offload::{PrefixCacheOffload, Transport, parse_size},
    runtime_package::LoadOptions,
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
    /// Serve an exported .safetensors package without Python.
    Run(Run),
    /// Accept rank zero's packages and executables and run its peer ranks on this host.
    Agent(AgentArgs),
    /// Manage remembered publisher trust.
    Trust {
        #[command(subcommand)]
        command: TrustCommand,
    },
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
    after_help = "Examples:\n  flare run model.safetensors --gpus 0 --bind 127.0.0.1:8000\n  flare run model.safetensors --embedding-table-storage ram --mtp-draft-tokens 3\n\nConcurrency and draft limits must fit the package's prepared capacities.\nOmitted concurrency and speculation options use package defaults."
)]
pub struct Run {
    /// Additional publisher verification profiles (JSON); does not grant execution trust.
    #[arg(long, value_name = "PATH", action = clap::ArgAction::Append)]
    pub publisher_keys: Vec<PathBuf>,
    /// Verify the complete stored allocation and table region before CUDA initialization.
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

    /// Path to the prepared .safetensors package.
    #[arg(value_name = "PACKAGE")]
    pub package: PathBuf,

    /// Address and port for the HTTP server.
    #[arg(
        long,
        value_name = "IP:PORT",
        default_value = "127.0.0.1:8000",
        help_heading = "Serving options"
    )]
    pub bind: SocketAddr,

    /// GPUs in rank order, relative to CUDA_VISIBLE_DEVICES when set.
    #[arg(long, value_name = "N[,N...]", value_delimiter = ',', action = clap::ArgAction::Set,
        default_value = "0", value_parser = clap::value_parser!(i32).range(0..),
        help_heading = "Serving options")]
    pub gpus: Vec<i32>,

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

    /// Storage for auxiliary embedding tables (PLE/Engram) [default: RAM on a discrete GPU
    /// when the tables take at most half of available memory, otherwise disk].
    #[arg(long, value_name = "MODE", help_heading = "Serving options",
        value_parser = PossibleValuesParser::new(["disk", "ram"])
            .map(|value| if value == "disk" { Storage::Disk } else { Storage::Ram }))]
    pub embedding_table_storage: Option<Storage>,

    /// Maximum drafts per round; 0 disables speculation [default: package setting].
    #[arg(long, value_name = "N", help_heading = "Speculation options")]
    pub mtp_draft_tokens: Option<usize>,

    /// Adapt verification width to acceptance and round cost [default: package setting].
    #[arg(
        long,
        overrides_with = "no_mtp_adaptive_verification",
        help_heading = "Speculation options"
    )]
    pub mtp_adaptive_verification: bool,

    /// Use a fixed verification width [default: package setting].
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

impl Run {
    /// Serving options every rank of a multi-host package must load with: they select the same
    /// graphs and capacities on every host.
    pub fn rank_options(&self) -> Vec<String> {
        let mut options = Vec::new();
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
        options
    }

    pub fn load_options(&self) -> LoadOptions {
        LoadOptions {
            embedding_table_storage: self.embedding_table_storage,
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
    fn omitted_limits_preserve_package_defaults() {
        let args = run(&["flare", "run", "model.safetensors"]);
        assert_eq!(args.bind.to_string(), "127.0.0.1:8000");
        let options = args.load_options();
        assert_eq!(options.device_labels, Some(vec![0]));
        assert_eq!(options.embedding_table_storage, None);
        assert_eq!(options.max_concurrency, None);
        assert_eq!(options.mtp_draft_tokens, None);
        assert_eq!(options.adaptive_verification, None);
        assert_eq!(options.prefix_cache, PrefixCacheOffload::default());
    }

    #[test]
    fn explicit_options_reach_the_loader() {
        let args = run(&[
            "flare",
            "run",
            "--gpus=1",
            "--gpus=2,0",
            "--bind",
            "[::1]:8001",
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
        assert_eq!(args.bind.to_string(), "[::1]:8001");
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
}
