//! Serve a prepared model package without a Python interpreter.

mod agent;
mod cli;
mod cluster;
mod environment;
mod launch;
mod trust;

use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

use clap::Parser;
use flare_helios::{
    model::{ModelContract, glm5next::Glm5Next, qwen4exp::Qwen4Exp},
    runtime_package::{NativeModel, VerifiedPackage},
    server::{Bootstrap, Server},
    startup::StartupProgress,
};

static STOP: AtomicBool = AtomicBool::new(false);
extern "C" fn stop(_: libc::c_int) {
    STOP.store(true, Ordering::Relaxed);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = match cli::Cli::parse().command {
        cli::Command::Run(args) => args,
        cli::Command::Agent(agent_args) => {
            let token = agent::read_token(
                &agent_args
                    .token_file
                    .unwrap_or_else(agent::default_token_path),
            )
            .map_err(|e| e.to_string())?;
            return agent::serve(&agent_args.listen, token, agent::default_cache())
                .map_err(|e| e.to_string().into());
        }
        cli::Command::Trust {
            command:
                cli::TrustCommand::Revoke {
                    fingerprint,
                    trust_store,
                },
        } => {
            return trust::revoke(&fingerprint, trust_store.as_deref());
        }
    };
    let root = args
        .package
        .canonicalize()
        .map_err(|error| format!("cannot access package {}: {error}", args.package.display()))?;
    let key = args
        .api_key_file
        .as_ref()
        .map(|path| {
            std::fs::read_to_string(path)
                .map(|value| value.trim().to_owned())
                .map_err(|error| format!("cannot read API key file {}: {error}", path.display()))
        })
        .transpose()?;
    let options = args.load_options();
    let rank_options = args.rank_options();
    let devices = &args.gpus;
    let inherited =
        std::env::var("CUDA_VISIBLE_DEVICES")
            .map(Some)
            .or_else(|error| match error {
                std::env::VarError::NotPresent => Ok(None),
                error => Err(error),
            })?;
    let visibility = environment::visibility(devices, inherited.as_deref())?;
    let local_devices: Vec<i32> = (0..devices.len() as i32).collect();
    // SAFETY: Startup is single-threaded here, before Helios starts workers or
    // loads CUDA. Packages contain compiled cubins and require no driver JIT.
    unsafe {
        std::env::set_var("CUDA_VISIBLE_DEVICES", visibility);
        std::env::set_var("CUDA_FORCE_PRELOAD_LIBRARIES", "0");
        std::env::set_var("CUDA_DISABLE_JIT", "1");
    }
    let mut progress = StartupProgress::stderr();
    let phase = progress.phase("Verify package", None);
    let package = VerifiedPackage::open(&root, &trust::publishers(&args.publisher_keys)?)?;
    let publisher = package.publisher().clone();
    phase.finish("signatures");
    let authorized = trust::authorize(package, &args, &mut progress)?;
    // Rank zero of a multi-host package starts its peers first, so every host loads in parallel.
    let mut rendezvous = args.rendezvous;
    let _peers = if args.hosts.is_empty() {
        None
    } else {
        let (world, export_id) = flare_helios::runtime_package::package_cluster(&authorized)
            .ok_or("--hosts applies only to a multi-host package")?;
        if args.hosts.len() + 1 != world {
            return Err(format!(
                "the package has {world} ranks; --hosts names {} peer hosts",
                args.hosts.len()
            )
            .into());
        }
        if args.rank.is_some_and(|rank| rank != 0) {
            return Err("--hosts starts peers from rank zero".into());
        }
        let listen = rendezvous.get_or_insert_with(|| "0.0.0.0:29800".into());
        let port = listen
            .rsplit_once(':')
            .map(|(_, port)| port)
            .ok_or("--rendezvous needs HOST:PORT")?;
        let address = launch::address_toward(&args.hosts[0]).map_err(|e| e.to_string())?;
        let token = agent::read_token(
            args.agent_token_file
                .as_deref()
                .unwrap_or(&agent::default_token_path()),
        )
        .map_err(|e| e.to_string())?;
        let phase = progress.phase("Start peer ranks", None);
        let peers = launch::start(
            &args.hosts,
            &root,
            &export_id,
            &format!("{address}:{port}"),
            &rank_options,
            &token,
            &publisher,
        )
        .map_err(|e| e.to_string())?;
        phase.finish(format!("{} at {address}:{port}", args.hosts.join(", ")));
        Some(peers)
    };
    // SAFETY: The trusted package owners remain live until every rank drains.
    let owner = unsafe { NativeModel::load(authorized, &local_devices, &options, &progress)? };
    let cluster = match (owner.cluster().cloned(), rendezvous) {
        (Some(cluster), Some(address)) => Some((cluster, address)),
        (Some(_), None) => {
            return Err("a multi-host package requires --rendezvous HOST:PORT".into());
        }
        (None, Some(_)) => {
            return Err("--rendezvous applies only to a multi-host package".into());
        }
        (None, None) => None,
    };
    if let Some((cluster, address)) = &cluster {
        if cluster.rank != 0 {
            return serve_peer(owner, cluster, address, progress);
        }
    }
    let phase = progress.phase("Read model configuration", None);
    let mut config = owner.server_config().clone();
    config["bind"] = args.bind.to_string().into();
    config["api_key"] = key.unwrap_or_default().into();
    phase.finish(config["model_id"].as_str().unwrap_or_default());
    match config["model_contract"]["schema"].as_str() {
        Some(flare_helios::model::qwen4exp::INPUT_SCHEMA) => {
            serve::<Qwen4Exp>(owner, config, progress, cluster)
        }
        Some(flare_helios::model::glm5next::INPUT_SCHEMA) => {
            serve::<Glm5Next>(owner, config, progress, cluster)
        }
        Some(flare_helios::model::deepseek4::INPUT_SCHEMA) => {
            serve::<flare_helios::model::deepseek4::DeepSeek4>(owner, config, progress, cluster)
        }
        Some(flare_helios::model::deepseek41::INPUT_SCHEMA) => {
            serve::<flare_helios::model::deepseek41::DeepSeek41>(owner, config, progress, cluster)
        }
        Some(flare_helios::model::mimo::INPUT_SCHEMA) => {
            serve::<flare_helios::model::mimo::MiMo>(owner, config, progress, cluster)
        }
        _ => Err("unsupported model contract".into()),
    }
}

/// Run a peer rank of a multi-host package: join rank zero and replay its coordinated steps until
/// rank zero drains every rank.
fn serve_peer(
    owner: NativeModel,
    cluster: &flare_helios::runtime_package::ClusterRank,
    address: &str,
    mut progress: StartupProgress,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut manifest = owner.manifests()?.remove(0);
    let joined = cluster::peer(&owner, cluster, address, &progress)?;
    manifest.collectives = Some(cluster::health(&joined.transport));
    progress.finish_animation();
    eprintln!(
        "Serving rank {} of {} for rank zero at {address}",
        cluster.rank, cluster.world
    );
    // SAFETY: The package owner and the transport stay live until rank zero confirms drain, and
    // are intentionally retained when it does not.
    let shutdown = unsafe {
        flare_helios::execution::serve_rank(manifest, joined.streams, cluster.rank, cluster.world)
    };
    if !shutdown.drained {
        std::mem::forget(joined.transport);
        std::mem::forget(owner);
        return Err("native replay did not drain".into());
    }
    // The proxy stops before the package releases the region it posts from.
    drop(joined.transport);
    drop(owner);
    match shutdown.error {
        Some(error) => Err(error.into()),
        None => Ok(()),
    }
}

fn serve<M: ModelContract>(
    owner: NativeModel,
    config: serde_json::Value,
    mut progress: StartupProgress,
    cluster: Option<(flare_helios::runtime_package::ClusterRank, String)>,
) -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: This executable loads a trusted package and retains its resources
    // until the native server confirms every replay stream has drained.
    let phase = progress.phase("Load tokenizer", None);
    let mut manifests = owner.manifests()?;
    let mut manifest = manifests.remove(0);
    let mut transport = None;
    let peers = match &cluster {
        Some((cluster, address)) => {
            let joined = cluster::rank_zero(&owner, cluster, address, &progress)?;
            manifest.collectives = Some(cluster::health(&joined.transport));
            transport = Some(joined.transport);
            Some(flare_helios::execution::RankPeers::Processes(
                joined.streams,
            ))
        }
        None => (!manifests.is_empty())
            .then_some(flare_helios::execution::RankPeers::Threads(manifests)),
    };
    let bootstrap = Bootstrap::load::<M>(&config.to_string(), &manifest)?;
    phase.finish(format!(
        "{} token vocabulary",
        bootstrap.model.contract.vocabulary
    ));
    let phase = progress.phase("Start server", None);
    let context_tokens = bootstrap.model.limits.max_sequence_tokens;
    let server = match unsafe {
        Server::<M>::start(
            manifest,
            bootstrap.model,
            bootstrap.frontend,
            bootstrap.images,
            bootstrap.server,
            peers,
        )
    } {
        Ok(server) => server,
        Err(error) => {
            if !error.drained {
                std::mem::forget(transport);
                std::mem::forget(owner);
            }
            return Err(error.message.into());
        }
    };
    phase.finish(format!(
        "http://{} · {context_tokens} tokens/request",
        server.address()
    ));
    unsafe {
        libc::signal(libc::SIGINT, stop as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, stop as *const () as libc::sighandler_t);
    }
    progress.finish_animation();
    if let Err(error) = server.start_dashboard() {
        eprintln!("Could not start dashboard: {error}");
    }
    while !STOP.load(Ordering::Relaxed) && server.wait(Duration::from_millis(250)) {}
    server.stop_dashboard();
    // Allow the terminal multiplexer to finish restoring the main screen.
    std::thread::sleep(Duration::from_millis(50));
    let phase = progress.phase("Shut down server", None);
    let shutdown = server.close();
    if !shutdown.drained {
        std::mem::forget(transport);
        std::mem::forget(owner);
        return Err("native replay did not drain".into());
    }
    if let Some(error) = shutdown.error {
        return Err(error.into());
    }
    if !STOP.load(Ordering::Relaxed) {
        return Err("native server stopped unexpectedly".into());
    }
    drop(server);
    // The proxy stops before the package releases the region it posts from.
    drop(transport);
    drop(owner);
    phase.finish("");
    Ok(())
}
