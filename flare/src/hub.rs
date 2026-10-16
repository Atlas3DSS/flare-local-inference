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

fn repository(text: &str) -> Result<(&str, Option<&str>)> {
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

fn choose_file(files: impl IntoIterator<Item = String>, requested: Option<&str>) -> Result<String> {
    if let Some(filename) = requested {
        validate_relative(filename)?;
        if !filename.ends_with(".safetensors") {
            return Err("--filename must name a prepared .safetensors package".into());
        }
        return Ok(filename.to_owned());
    }
    let mut candidates: Vec<_> = files
        .into_iter()
        .filter(|file| file.ends_with(".safetensors"))
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
    let token = options
        .token_file
        .as_ref()
        .map(|path| {
            let value = std::fs::read_to_string(path)
                .map_err(|error| format!("cannot read token file {}: {error}", path.display()))?;
            let token = value.trim();
            if token.is_empty() {
                return Err("Hugging Face token file is empty".to_owned());
            }
            Ok(token.to_owned())
        })
        .transpose()?;
    let api = ApiBuilder::from_cache(cache.clone())
        .with_token(token)
        .with_endpoint("https://huggingface.co".into())
        .with_progress(std::io::stderr().is_terminal())
        .with_retries(3)
        .build()?;
    fetch(&api, &cache, &repo, options.filename.as_deref())
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
        api.repo(pinned).download(&filename)?
    };
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
    let path = directory.join(filename);
    if !path.is_file() {
        return Err(format!("package is not cached: {}", path.display()).into());
    }
    Ok(path)
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
        std::fs::write(directory.join("model.safetensors"), b"package fixture").unwrap();
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

    fn mock_hub(responses: Vec<(String, String)>) -> (String, thread::JoinHandle<()>) {
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
                socket.write_all(response.as_bytes()).unwrap();
            }
        });
        (endpoint, worker)
    }

    fn metadata_response() -> String {
        let body =
            format!(r#"{{"sha":"{COMMIT}","siblings":[{{"rfilename":"model.safetensors"}}]}}"#);
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
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
        let (endpoint, worker) = mock_hub(vec![
            ("/api/models/owner/model/revision/gb202".into(), metadata_response()),
            (resolve.clone(), format!("HTTP/1.1 206 Partial Content\r\nx-repo-commit: {COMMIT}\r\netag: fixture\r\nContent-Range: bytes 0-0/7\r\nContent-Length: 1\r\nConnection: close\r\n\r\np")),
            (resolve, "HTTP/1.1 206 Partial Content\r\nContent-Length: 7\r\nConnection: close\r\n\r\npackage".into()),
        ]);
        let api = test_api(&cache, endpoint);
        let path = fetch(&api, &cache, &repo, None).unwrap();
        worker.join().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"package");
        assert_eq!(cached(&cache, &repo, None).unwrap(), path);
    }

    #[test]
    fn online_cache_hit_resolves_revision_without_redownloading_weights() {
        let temp = tempfile::tempdir().unwrap();
        let cache = Cache::new(temp.path().to_owned());
        let repo = Repo::with_revision("owner/model".into(), RepoType::Model, "gb10".into());
        let directory = snapshot(&cache, &repo, COMMIT);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("model.safetensors"), b"existing package").unwrap();
        let (endpoint, worker) = mock_hub(vec![(
            "/api/models/owner/model/revision/gb10".into(),
            metadata_response(),
        )]);
        let path = fetch(&test_api(&cache, endpoint), &cache, &repo, None).unwrap();
        worker.join().unwrap();
        assert_eq!(std::fs::read(path).unwrap(), b"existing package");
    }
}
