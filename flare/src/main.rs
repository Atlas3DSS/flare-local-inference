//! Serve a prepared model package without a Python interpreter.

mod cli;
mod environment;

use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

use clap::Parser;
use flare_helios::{
    model::{ModelContract, glm5next::Glm5Next, qwen4exp::Qwen4Exp},
    runtime_package::NativeModel,
    server::{Bootstrap, Server},
    startup::StartupProgress,
};

static STOP: AtomicBool = AtomicBool::new(false);
extern "C" fn stop(_: libc::c_int) {
    STOP.store(true, Ordering::Relaxed);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli::Command::Run(args) = cli::Cli::parse().command;
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
    let devices = args.gpus;
    let inherited =
        std::env::var("CUDA_VISIBLE_DEVICES")
            .map(Some)
            .or_else(|error| match error {
                std::env::VarError::NotPresent => Ok(None),
                error => Err(error),
            })?;
    let visibility = environment::visibility(&devices, inherited.as_deref())?;
    let local_devices: Vec<i32> = (0..devices.len() as i32).collect();
    // SAFETY: Startup is single-threaded here, before Helios starts workers or
    // loads CUDA. Packages contain compiled cubins and require no driver JIT.
    unsafe {
        std::env::set_var("CUDA_VISIBLE_DEVICES", visibility);
        std::env::set_var("CUDA_FORCE_PRELOAD_LIBRARIES", "0");
        std::env::set_var("CUDA_DISABLE_JIT", "1");
    }
    let progress = StartupProgress::stderr();
    // SAFETY: The trusted package owners remain live until every rank drains.
    let owner = unsafe { NativeModel::load(&root, &local_devices, &options, &progress)? };
    let phase = progress.phase("Read model configuration", None);
    let mut config = owner.server_config().clone();
    config["bind"] = args.bind.to_string().into();
    config["api_key"] = key.unwrap_or_default().into();
    phase.finish(config["model_id"].as_str().unwrap_or_default());
    match config["model_contract"]["schema"].as_str() {
        Some(flare_helios::model::qwen4exp::INPUT_SCHEMA) => {
            serve::<Qwen4Exp>(owner, config, progress)
        }
        Some(flare_helios::model::glm5next::INPUT_SCHEMA) => {
            serve::<Glm5Next>(owner, config, progress)
        }
        Some(flare_helios::model::deepseek41::INPUT_SCHEMA) => {
            serve::<flare_helios::model::deepseek41::DeepSeek41>(owner, config, progress)
        }
        _ => Err("unsupported model contract".into()),
    }
}

fn serve<M: ModelContract>(
    owner: NativeModel,
    config: serde_json::Value,
    mut progress: StartupProgress,
) -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: This executable loads a trusted package and retains its resources
    // until the native server confirms every replay stream has drained.
    let phase = progress.phase("Load tokenizer", None);
    let mut manifests = owner.manifests()?;
    let manifest = manifests.remove(0);
    let peers =
        (!manifests.is_empty()).then_some(flare_helios::execution::RankPeers::Threads(manifests));
    let bootstrap = Bootstrap::load::<M>(&config.to_string(), &manifest)?;
    phase.finish(format!(
        "{} token vocabulary",
        bootstrap.model.contract.vocabulary
    ));
    let phase = progress.phase("Start server", None);
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
                std::mem::forget(owner);
            }
            return Err(error.message.into());
        }
    };
    phase.finish(format!("http://{}", server.address()));
    unsafe {
        libc::signal(libc::SIGINT, stop as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, stop as *const () as libc::sighandler_t);
    }
    progress.finish_animation();
    if let Err(error) = server.start_dashboard() {
        eprintln!("Could not start dashboard: {error}");
    }
    while !STOP.load(Ordering::Relaxed) && server.wait(Duration::from_millis(250)) {}
    let shutdown = server.close();
    if !shutdown.drained {
        std::mem::forget(owner);
        return Err("native replay did not drain".into());
    }
    if let Some(error) = shutdown.error {
        return Err(error.into());
    }
    if !STOP.load(Ordering::Relaxed) {
        return Err("native server stopped unexpectedly".into());
    }
    Ok(())
}
