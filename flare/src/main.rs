//! Serve a prepared model package without a Python interpreter.

use std::{
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

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
    let mut arguments = std::env::args().skip(1);
    let Some(command) = arguments.next() else {
        return Err(
            "usage: flare run PACKAGE [--bind IP:PORT] [--gpus N[,N...]] [--api-key-file PATH]"
                .into(),
        );
    };
    if command != "run" {
        return Err("expected flare run PACKAGE".into());
    }
    let root =
        PathBuf::from(arguments.next().ok_or("package directory is required")?).canonicalize()?;
    let (mut bind, mut key, mut devices) = ("127.0.0.1:8000".to_owned(), None, vec![0]);
    while let Some(flag) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
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
    let mut progress = StartupProgress::stderr();
    // SAFETY: The trusted package owners remain live until every rank drains.
    let owner = unsafe { NativeModel::load(&root, &devices, &progress)? };
    let root = owner.frontend_root();
    let phase = progress.phase("Read model configuration", None);
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("server.json"))?)?;
    let schema = config["model_contract"]["schema"]
        .as_str()
        .ok_or("model_contract.schema is required")?
        .to_owned();
    config["bind"] = bind.into();
    config["api_key"] = key.unwrap_or_default().into();
    for asset in config["assets"]
        .as_object_mut()
        .ok_or("missing frontend assets")?
        .values_mut()
    {
        let file = asset["path"]
            .as_str()
            .ok_or("invalid frontend asset path")?;
        if std::path::Path::new(file).components().count() != 1 {
            return Err("frontend assets must be package-local files".into());
        }
        asset["path"] = root
            .join(file)
            .to_str()
            .ok_or("non-UTF8 package path")?
            .into();
    }
    phase.finish(config["model_id"].as_str().unwrap_or_default());
    match schema.as_str() {
        flare_helios::model::qwen4exp::INPUT_SCHEMA => {
            serve::<Qwen4Exp>(owner, &config, &mut progress)
        }
        flare_helios::model::glm5next::INPUT_SCHEMA => {
            serve::<Glm5Next>(owner, &config, &mut progress)
        }
        _ => Err(format!("unsupported model contract {schema:?}").into()),
    }
}

/// Start the native server for one model contract and block until it stops and drains.
fn serve<M: ModelContract>(
    owner: NativeModel,
    config: &serde_json::Value,
    progress: &mut StartupProgress,
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
