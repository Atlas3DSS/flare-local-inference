//! Download prepared packages from hardware-specific Hugging Face revisions.

use std::{
    io::IsTerminal,
    path::{Path, PathBuf},
};

use hf_hub::{
    Cache, Repo, RepoType,
    api::sync::{Api, ApiBuilder},
};

use crate::cli::HubOptions;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn is_repository(path: &Path) -> bool {
    !path.exists()
        && !path.is_absolute()
        && !path.starts_with(".")
        && !path.starts_with("..")
        && path.to_str().is_some_and(|text| {
            !text.contains('\\') && !text.ends_with(".safetensors") && repository(text).is_ok()
        })
}

pub(crate) fn repository(text: &str) -> Result<(&str, Option<&str>)> {
    let (id, revision) = match text.split_once('@') {
        Some((id, revision)) if !revision.is_empty() => (id, Some(revision)),
        Some(_) => return Err("Hugging Face revision cannot be empty".into()),
        None => (text, None),
    };
    let parts: Vec<_> = id.split('/').collect();
    if parts.len() != 2
        || parts.iter().any(|part| {
            part.is_empty()
                || part.starts_with(['.', '-'])
                || part.ends_with(['.', '-'])
                || part.contains("..")
                || part.contains("--")
                || !part
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c))
        })
    {
        return Err("expected a Hugging Face repository as ORG/NAME[@REVISION]".into());
    }
    if let Some(revision) = revision {
        validate_relative(revision)?;
    }
    Ok((id, revision))
}

fn validate_relative(value: &str) -> Result<()> {
    if value.is_empty()
        || value.contains(['\\', '\0'])
        || value
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == ".." || part.contains(':'))
    {
        return Err(format!("invalid repository-relative path or revision: {value:?}").into());
    }
    Ok(())
}

fn is_companion(filename: &str) -> bool {
    let Some(stem) = filename.strip_suffix(".safetensors") else {
        return false;
    };
    let Some((_, numbers)) = stem.rsplit_once("-data-") else {
        return false;
    };
    let bytes = numbers.as_bytes();
    bytes.len() == 8
        && &bytes[2..6] == b"-of-"
        && bytes[..2].iter().chain(&bytes[6..]).all(u8::is_ascii_digit)
}

fn choose_file(files: impl IntoIterator<Item = String>, requested: Option<&str>) -> Result<String> {
    if let Some(filename) = requested {
        validate_relative(filename)?;
        if !filename.ends_with(".safetensors") || is_companion(filename) {
            return Err("--filename must name a prepared .safetensors package".into());
        }
        return Ok(filename.to_owned());
    }
    let mut candidates: Vec<_> = files
        .into_iter()
        .filter(|file| file.ends_with(".safetensors") && !is_companion(file))
        .collect();
    candidates.sort();
    match candidates.as_slice() {
        [filename] => {
            validate_relative(filename)?;
            Ok(filename.clone())
        }
        [] => Err("revision contains no .safetensors package; choose a revision containing prepared Flare packages".into()),
        _ => Err(format!("revision contains several .safetensors files; select one with --filename:\n  {}", candidates.join("\n  ")).into()),
    }
}

pub fn download(source: &str, options: &HubOptions, gpu_count: usize) -> Result<PathBuf> {
    let (id, given_revision) = repository(source)?;
    let revision = match options.revision.as_deref().or(given_revision) {
        Some(revision) => revision.to_owned(),
        None => crate::hardware::detect(gpu_count)?.to_owned(),
    };
    validate_relative(&revision)?;
    if let Some(filename) = &options.filename {
        choose_file([], Some(filename))?;
    }
    let cache_dir = match &options.cache_dir {
        Some(path) => std::path::absolute(path)?,
        None => crate::environment::home_directory()?.join(".cache/huggingface/hub"),
    };
    let cache = Cache::new(cache_dir);
    let repo = Repo::with_revision(id.to_owned(), RepoType::Model, revision.clone());
    eprintln!("Package source: {id}@{revision}");
    if options.offline {
        return cached(&cache, &repo, options.filename.as_deref());
    }
    let api = ApiBuilder::from_cache(cache.clone())
        .with_token(token(options)?)
        .with_endpoint("https://huggingface.co".into())
        .with_progress(std::io::stderr().is_terminal())
        .with_retries(3)
        .build()?;
    fetch(&api, &cache, &repo, options.filename.as_deref())
}

/// The access token named by `--token-file`, for private repositories.
pub(crate) fn token(options: &HubOptions) -> Result<Option<String>> {
    options
        .token_file
        .as_ref()
        .map(|path| {
            let value = std::fs::read_to_string(path)
                .map_err(|error| format!("cannot read token file {}: {error}", path.display()))?;
            let token = value.trim();
            if token.is_empty() {
                return Err("Hugging Face token file is empty".into());
            }
            Ok(token.to_owned())
        })
        .transpose()
}

fn fetch(api: &Api, cache: &Cache, repo: &Repo, requested: Option<&str>) -> Result<PathBuf> {
    let info = api.repo(repo.clone()).info()?;
    if info.sha.len() != 40 || !info.sha.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("Hugging Face returned an invalid commit ID".into());
    }
    let filename = choose_file(
        info.siblings.into_iter().map(|file| file.rfilename),
        requested,
    )?;
    let pinned = Repo::with_revision(repo.url(), RepoType::Model, info.sha.clone());
    let path = snapshot(cache, &pinned, &info.sha).join(&filename);
    eprintln!("Resolved package: {}@{} / {filename}", repo.url(), info.sha);
    let path = if path.is_file() {
        path
    } else {
        eprintln!("Downloading model: {filename}...");
        api.repo(pinned.clone()).download(&filename)?
    };
    let companions = flare_helios::runtime_package::package_companions(&path)?;
    let parent = Path::new(&filename).parent().unwrap();
    for name in &companions {
        let relative = parent.join(name).to_string_lossy().replace('\\', "/");
        if !snapshot(cache, &pinned, &info.sha)
            .join(&relative)
            .is_file()
        {
            eprintln!("Downloading model data: {relative}...");
            api.repo(pinned.clone()).download(&relative)?;
        }
    }
    let path = package_view(cache, repo, &info.sha, &filename, &companions)?;
    cache.repo(repo.clone()).create_ref(&info.sha)?;
    Ok(path)
}

fn snapshot(cache: &Cache, repo: &Repo, commit: &str) -> PathBuf {
    cache
        .path()
        .join(repo.folder_name())
        .join("snapshots")
        .join(commit)
}

fn cached(cache: &Cache, repo: &Repo, requested: Option<&str>) -> Result<PathBuf> {
    let revision = repo.revision();
    let commit =
        if revision.len() == 40 && revision.bytes().all(|c| c.is_ascii_hexdigit()) {
            revision.to_owned()
        } else {
            let path = cache
                .path()
                .join(repo.folder_name())
                .join("refs")
                .join(revision);
            std::fs::read_to_string(&path).map_err(|error| {
            format!("cannot resolve cached revision {revision}: {error}; download it online first")
        })?.trim().to_owned()
        };
    if commit.len() != 40 || !commit.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("cached revision contains an invalid commit ID".into());
    }
    let directory = snapshot(cache, repo, &commit);
    let filename = if let Some(filename) = requested {
        choose_file([], Some(filename))?
    } else {
        let mut files = Vec::new();
        collect_files(&directory, &directory, &mut files)?;
        choose_file(files, None)?
    };
    let path = directory.join(&filename);
    if !path.is_file() {
        return Err(format!("package is not cached: {}", path.display()).into());
    }
    let companions = flare_helios::runtime_package::package_companions(&path)?;
    package_view(cache, repo, &commit, &filename, &companions)
}

/// Snapshot entries may be symlinks to blobs. A package view uses regular hardlinks so the
/// runtime can reject symlink companions without duplicating model bytes.
fn package_view(
    cache: &Cache,
    repo: &Repo,
    commit: &str,
    filename: &str,
    companions: &[String],
) -> Result<PathBuf> {
    let primary = snapshot(cache, repo, commit).join(filename);
    if companions.is_empty() {
        return Ok(primary);
    }
    let view = cache
        .path()
        .join(repo.folder_name())
        .join("flare-packages")
        .join(commit)
        .join(filename);
    std::fs::create_dir_all(&view)?;
    let basename = primary
        .file_name()
        .ok_or("missing package filename")?
        .to_string_lossy()
        .into_owned();
    for name in companions.iter().chain(std::iter::once(&basename)) {
        let source = primary
            .parent()
            .unwrap()
            .join(name)
            .canonicalize()
            .map_err(|e| format!("package file is not cached: {name}: {e}"))?;
        if !source.is_file() {
            return Err(format!("package file is not regular: {name}").into());
        }
        let target = view.join(name);
        match std::fs::hard_link(&source, &target) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                let info = std::fs::symlink_metadata(&target)?;
                if !info.is_file() || info.len() != std::fs::metadata(&source)?.len() {
                    return Err(format!("invalid cached package view: {}", target.display()).into());
                }
            }
            Err(e) => {
                return Err(format!(
                    "cannot hardlink package cache view {}: {e}",
                    target.display()
                )
                .into());
            }
        }
    }
    Ok(view.join(basename))
}

fn collect_files(root: &Path, directory: &Path, files: &mut Vec<String>) -> Result<()> {
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            collect_files(root, &entry.path(), files)?;
        } else if entry.path().is_file() {
            files.push(
                entry
                    .path()
                    .strip_prefix(root)?
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{BufRead, BufReader, Write},
        net::TcpListener,
        thread,
        time::Duration,
    };

    const COMMIT: &str = "0123456789012345678901234567890123456789";

    fn fixture(companions: &[&str]) -> Vec<u8> {
        fn bytes(field: u8, body: &[u8]) -> Vec<u8> {
            let mut out = vec![field * 8 + 2];
            let mut n = body.len();
            while n >= 128 {
                out.push((n as u8 & 127) | 128);
                n >>= 7;
            }
            out.push(n as u8);
            out.extend_from_slice(body);
            out
        }
        // ProtectedIndex -> PublisherMetadata -> ModelIndex, with unsigned discovery metadata.
        let index = [8, if companions.is_empty() { 4 } else { 5 }];
        let metadata = bytes(1, &index);
        let mut envelope = vec![8, 4];
        for (kind, body) in [(1, metadata), (2, vec![]), (3, vec![])] {
            envelope.extend(bytes(4, &[vec![8, kind], bytes(2, &body)].concat()));
        }
        let mut data = Vec::new();
        for name in companions {
            data.extend(bytes(4, &bytes(3, name.as_bytes())));
        }
        let rank = [bytes(2, &[]), bytes(5, &data)].concat();
        envelope.extend(bytes(5, &[vec![8, 4], bytes(2, &rank)].concat()));
        let header = serde_json::json!({
            "__metadata__": {"format": "flare"},
            "flare.payload": {"dtype": "U8", "shape": [0], "data_offsets": [0, 0]},
            "flare.metadata": {"dtype": "U8", "shape": [envelope.len()], "data_offsets": [0, envelope.len()]},
        });
        let mut result = 4088u64.to_le_bytes().to_vec();
        result.extend(serde_json::to_vec(&header).unwrap());
        result.resize(4096, b' ');
        result.extend(envelope);
        result
    }

    #[test]
    fn local_paths_and_repository_revisions_are_distinct() {
        assert!(is_repository(Path::new("owner/model")));
        assert!(is_repository(Path::new("owner/model@release/gb202")));
        for path in [
            "model.safetensors",
            "dir/model.safetensors",
            "./owner/model",
            "/owner/model",
            "../owner/model",
            "C:\\models\\model.safetensors",
        ] {
            assert!(!is_repository(Path::new(path)), "{path}");
        }
        assert_eq!(
            repository("owner/model@gb10").unwrap(),
            ("owner/model", Some("gb10"))
        );
        for source in [
            "owner",
            "owner/model/extra",
            "../model",
            "owner/model@",
            "owner/model@../../other",
        ] {
            assert!(repository(source).is_err(), "{source}");
        }
    }

    #[test]
    fn package_selection_rejects_ambiguous_and_escaping_filenames() {
        assert_eq!(
            choose_file(["README.md".into(), "model.safetensors".into()], None).unwrap(),
            "model.safetensors"
        );
        let error = choose_file(["tp4.safetensors".into(), "tp2.safetensors".into()], None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("--filename"));
        assert!(error.contains("tp2.safetensors\n  tp4.safetensors"));
        for file in [
            "../model.safetensors",
            "/model.safetensors",
            "C:\\model.safetensors",
            "weights.json",
        ] {
            assert!(choose_file([], Some(file)).is_err(), "{file}");
        }
    }

    #[test]
    fn offline_cache_reads_python_style_snapshots_without_commit_refs() {
        let temp = tempfile::tempdir().unwrap();
        let cache = Cache::new(temp.path().to_owned());
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, "gb10".into());
        let directory = snapshot(&cache, &repo, COMMIT);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("model.safetensors"), &fixture(&[])).unwrap();
        cache.repo(repo.clone()).create_ref(COMMIT).unwrap();
        assert_eq!(
            cached(&cache, &repo, None).unwrap(),
            directory.join("model.safetensors")
        );
        let pinned = Repo::with_revision(repo.url(), RepoType::Model, COMMIT.into());
        assert_eq!(
            cached(&cache, &pinned, None).unwrap(),
            directory.join("model.safetensors")
        );
        assert!(cached(&cache, &repo, Some("missing.safetensors")).is_err());
        std::fs::write(directory.join("another.safetensors"), b"other").unwrap();
        assert!(
            cached(&cache, &repo, None)
                .unwrap_err()
                .to_string()
                .contains("--filename")
        );
        cache
            .repo(repo.clone())
            .create_ref("../../outside")
            .unwrap();
        assert!(
            cached(&cache, &repo, None)
                .unwrap_err()
                .to_string()
                .contains("invalid commit")
        );
    }

    fn mock_hub(responses: Vec<(String, Vec<u8>)>) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let worker = thread::spawn(move || {
            for (expected, response) in responses {
                let (mut socket, _) = listener.accept().unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut reader = BufReader::new(&mut socket);
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                assert!(line.starts_with(&format!("GET {expected} ")), "{line}");
                loop {
                    line.clear();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" {
                        break;
                    }
                }
                socket.write_all(&response).unwrap();
            }
        });
        (endpoint, worker)
    }

    fn metadata_response() -> Vec<u8> {
        let body =
            format!(r#"{{"sha":"{COMMIT}","siblings":[{{"rfilename":"model.safetensors"}}]}}"#);
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        ).into_bytes()
    }

    fn test_api(cache: &Cache, endpoint: String) -> Api {
        ApiBuilder::from_cache(cache.clone())
            .with_token(None)
            .with_endpoint(endpoint)
            .with_progress(false)
            .build()
            .unwrap()
    }

    #[test]
    fn download_resolves_hardware_branch_then_fetches_the_exact_commit() {
        let temp = tempfile::tempdir().unwrap();
        let cache = Cache::new(temp.path().to_owned());
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, "gb202".into());
        let resolve = format!("/owner/model/resolve/{COMMIT}/model.safetensors");
        let package = fixture(&[]);
        let (endpoint, worker) = mock_hub(vec![
            ("/api/models/owner/model/revision/gb202".into(), metadata_response()),
            (resolve.clone(), format!("HTTP/1.1 206 Partial Content\r\nx-repo-commit: {COMMIT}\r\netag: fixture\r\nContent-Range: bytes 0-0/{}\r\nContent-Length: 1\r\nConnection: close\r\n\r\np", package.len()).into_bytes()),
            (resolve, [format!("HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", package.len()).into_bytes(), package.clone()].concat()),
        ]);
        let api = test_api(&cache, endpoint);
        let path = fetch(&api, &cache, &repo, None).unwrap();
        worker.join().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), package);
        assert_eq!(cached(&cache, &repo, None).unwrap(), path);
    }

    #[test]
    fn online_cache_hit_resolves_revision_without_redownloading_weights() {
        let temp = tempfile::tempdir().unwrap();
        let cache = Cache::new(temp.path().to_owned());
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, "gb10".into());
        let directory = snapshot(&cache, &repo, COMMIT);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("model.safetensors"), &fixture(&[])).unwrap();
        let (endpoint, worker) = mock_hub(vec![(
            "/api/models/owner/model/revision/gb10".into(),
            metadata_response(),
        )]);
        let path = fetch(&test_api(&cache, endpoint), &cache, &repo, None).unwrap();
        worker.join().unwrap();
        assert_eq!(std::fs::read(path).unwrap(), fixture(&[]));
    }
    #[test]
    fn offline_shards_require_complete_snapshots_and_use_regular_file_views() {
        let temp = tempfile::tempdir().unwrap();
        let cache = Cache::new(temp.path().to_owned());
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, COMMIT.into());
        let directory = snapshot(&cache, &repo, COMMIT);
        std::fs::create_dir_all(&directory).unwrap();
        let name = "model-data-01-of-01.safetensors";
        std::fs::write(directory.join("model.safetensors"), fixture(&[name])).unwrap();
        assert!(
            cached(&cache, &repo, None)
                .unwrap_err()
                .to_string()
                .contains("not cached")
        );
        let blob = temp.path().join("blob");
        std::fs::write(&blob, b"companion data").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&blob, directory.join(name)).unwrap();
        #[cfg(windows)]
        std::fs::hard_link(&blob, directory.join(name)).unwrap();
        let path = cached(&cache, &repo, None).unwrap();
        assert_ne!(path, directory.join("model.safetensors"));
        assert!(
            std::fs::symlink_metadata(path.parent().unwrap().join(name))
                .unwrap()
                .is_file()
        );
        assert_eq!(
            std::fs::read(path.parent().unwrap().join(name)).unwrap(),
            b"companion data"
        );
        assert_eq!(cached(&cache, &repo, None).unwrap(), path);
        assert_eq!(
            choose_file(["model.safetensors".into(), name.into()], None).unwrap(),
            "model.safetensors"
        );
        assert!(choose_file([], Some(name)).is_err());
    }
    #[test]
    fn companions_download_from_the_primary_commit() {
        let temp = tempfile::tempdir().unwrap();
        let cache = Cache::new(temp.path().to_owned());
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, "gb202".into());
        let name = "model-data-01-of-01.safetensors";
        let directory = snapshot(&cache, &repo, COMMIT);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("model.safetensors"), fixture(&[name])).unwrap();
        let resolve = format!("/owner/model/resolve/{COMMIT}/{name}");
        let (endpoint, worker) = mock_hub(vec![
            ("/api/models/owner/model/revision/gb202".into(), metadata_response()),
            (resolve.clone(), format!("HTTP/1.1 206 Partial Content\r\nx-repo-commit: {COMMIT}\r\netag: data-fixture\r\nContent-Range: bytes 0-0/4\r\nContent-Length: 1\r\nConnection: close\r\n\r\nd").into_bytes()),
            (resolve, b"HTTP/1.1 206 Partial Content\r\nContent-Length: 4\r\nConnection: close\r\n\r\ndata".to_vec()),
        ]);
        let path = fetch(&test_api(&cache, endpoint), &cache, &repo, None).unwrap();
        worker.join().unwrap();
        assert_eq!(
            std::fs::read(path.parent().unwrap().join(name)).unwrap(),
            b"data"
        );
        assert_eq!(cached(&cache, &repo, None).unwrap(), path);
    }
}
