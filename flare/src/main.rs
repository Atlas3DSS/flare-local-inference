//! Serve a prepared model package without a Python interpreter.

mod environment;

use std::{
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

use flare_helios::{
    model::{ModelContract, glm5next::Glm5Next, qwen4exp::Qwen4Exp},
    runtime_package::{LoadOptions, NativeModel},
    server::{Bootstrap, Server},
    startup::StartupProgress,
};

static STOP: AtomicBool = AtomicBool::new(false);
extern "C" fn stop(_: libc::c_int) {
    STOP.store(true, Ordering::Relaxed);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let Some(command) = arguments.next() else {
        return Err(
            "usage: flare run PACKAGE [--bind IP:PORT] [--gpus N[,N...]] [--api-key-file PATH] [--max-concurrency N] [--mtp-draft-tokens N] [--mtp-adaptive-verification|--no-mtp-adaptive-verification] [--ngram-storage disk|ram]"
                .into(),
        );
    };
    if command != "run" {
        return Err("expected flare run PACKAGE".into());
    }
    let root = PathBuf::from(arguments.next().ok_or("package file is required")?).canonicalize()?;
    let (mut bind, mut key, mut devices) = ("127.0.0.1:8000".to_owned(), None, vec![0]);
    let mut options = LoadOptions::default();
    while let Some(flag) = arguments.next() {
        match flag.as_str() {
            "--mtp-adaptive-verification" => {
                options.adaptive_verification = Some(true);
                continue;
            }
            "--no-mtp-adaptive-verification" => {
                options.adaptive_verification = Some(false);
                continue;
            }
            _ => {}
        }
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--ngram-storage" => {
                options.ngram_storage = match value.as_str() {
                    "disk" => flare_helios::engram::Storage::Disk,
                    "ram" => flare_helios::engram::Storage::Ram,
                    _ => return Err("--ngram-storage must be disk or ram".into()),
                }
            }
            "--max-concurrency" => options.max_concurrency = Some(value.parse()?),
            "--mtp-draft-tokens" => options.mtp_draft_tokens = Some(value.parse()?),
            "--bind" => bind = value,
            "--gpus" => {
                devices = value
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<Vec<i32>, _>>()?
            }
            "--api-key-file" => key = Some(std::fs::read_to_string(value)?.trim().to_owned()),
            _ => return Err(format!("unknown option {flag}").into()),
        }
    }
    let inherited =
        std::env::var("CUDA_VISIBLE_DEVICES")
            .map(Some)
            .or_else(|error| match error {
                std::env::VarError::NotPresent => Ok(None),
                error => Err(error),
            })?;
    let visibility = environment::visibility(&devices, inherited.as_deref())?;
    let local_devices: Vec<i32> = (0..devices.len() as i32).collect();
    options.device_labels = Some(devices);
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
    config["bind"] = bind.into();
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
