#!/usr/bin/env python3
"""Build a signed binary APT repository in a previously absent directory."""

import argparse
import gzip
import hashlib
import lzma
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def run(*args, cwd=None):
    return subprocess.check_output(args, cwd=cwd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gnupg-home", type=Path, required=True)
    parser.add_argument("--key", required=True, help="full signing-key fingerprint")
    parser.add_argument("packages", type=Path, nargs="+")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9A-Fa-f]{40}|[0-9A-Fa-f]{64}", args.key):
        parser.error("--key must be a full OpenPGP fingerprint")
    output = args.output.resolve()
    if output.exists():
        parser.error("--output must not exist; build a staging tree before publication")
    output.parent.mkdir(parents=True, exist_ok=True)
    gpg = ["gpg", "--homedir", str(args.gnupg_home.resolve()), "--batch"]
    with tempfile.TemporaryDirectory(prefix=".flare-apt-", dir=output.parent) as temporary:
        root = Path(temporary)
        pool = root / "pool/main/f/flare-inference"
        pool.mkdir(parents=True)
        architectures = set()
        for package in args.packages:
            name, version, arch = run(
                "dpkg-deb", "--show", "--showformat=${Package}\n${Version}\n${Architecture}",
                str(package),
            ).decode().splitlines()
            if name != "flare-inference" or arch not in {"arm64", "amd64"}:
                parser.error(f"unsupported package: {name}, {arch}")
            if not re.fullmatch(r"[0-9][A-Za-z0-9.+:~\-]*", version):
                parser.error("invalid package version")
            destination = pool / f"{name}_{version}_{arch}.deb"
            if destination.exists():
                parser.error(f"duplicate package version and architecture: {destination.name}")
            shutil.copyfile(package, destination)
            architectures.add(arch)
        suite = root / "dists/noble"
        indices = []
        for arch in sorted(architectures):
            directory = suite / f"main/binary-{arch}"
            directory.mkdir(parents=True)
            packages = run("apt-ftparchive", "--arch", arch, "packages", "pool", cwd=root)
            for name, data in (
                ("Packages", packages),
                ("Packages.gz", gzip.compress(packages, mtime=0)),
                ("Packages.xz", lzma.compress(packages)),
            ):
                path = directory / name
                path.write_bytes(data)
                indices.append(path)
        release = run(
            "apt-ftparchive",
            "-o", "APT::FTPArchive::Release::Origin=Local Inference Lab",
            "-o", "APT::FTPArchive::Release::Label=Flare",
            "-o", "APT::FTPArchive::Release::Suite=noble",
            "-o", "APT::FTPArchive::Release::Codename=noble",
            "-o", f'APT::FTPArchive::Release::Architectures={" ".join(sorted(architectures))}',
            "-o", "APT::FTPArchive::Release::Components=main",
            "-o", "APT::FTPArchive::Release::Acquire-By-Hash=yes",
            "-o", "APT::FTPArchive::Release::Description=Flare native inference for Ubuntu 24.04",
            "release", ".", cwd=suite,
        )
        (suite / "Release").write_bytes(release)
        for path in indices:
            for algorithm in ("SHA256", "SHA512"):
                digest = hashlib.new(algorithm.lower(), path.read_bytes()).hexdigest()
                target = path.parent / "by-hash" / algorithm / digest
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        for flags, name in ((["--clearsign"], "InRelease"), (["--armor", "--detach-sign"], "Release.gpg")):
            subprocess.run([
                *gpg, "--local-user", args.key, "--digest-algo", "SHA256",
                "--output", str(suite / name), *flags, str(suite / "Release"),
            ], check=True)
        (root / "lil-flare-archive.asc").write_bytes(run(*gpg, "--armor", "--export", args.key))
        (root / "flare.sources").write_text(
            "Types: deb\nURIs: https://ppa.local-inference-lab.ai\nSuites: noble\n"
            f'Components: main\nArchitectures: {" ".join(sorted(architectures))}\n'
            "Signed-By: /etc/apt/keyrings/lil-flare-archive.asc\n"
        )
        subprocess.run([*gpg, "--verify", str(suite / "InRelease")], check=True)
        subprocess.run([*gpg, "--verify", str(suite / "Release.gpg"), str(suite / "Release")], check=True)
        # TemporaryDirectory creates a private root; the public tree needs traversal.
        root.chmod(0o755)
        root.rename(output)
    print(output)


if __name__ == "__main__":
    main()
