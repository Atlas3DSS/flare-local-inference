"""Include dependency license and notice files without shipping crate sources."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

destination = Path(sys.argv[1])
metadata = json.loads(subprocess.check_output([
    "cargo", "metadata", "--locked", "--format-version", "1",
    "--manifest-path", "flare/Cargo.toml",
]))
destination.mkdir(parents=True)
for package in metadata["packages"]:
    if package["source"] is None:
        continue
    root = Path(package["manifest_path"]).parent
    target = destination / f'{package["name"]}-{package["version"]}'
    target.mkdir()
    (target / "package.txt").write_text(
        f'{package["name"]} {package["version"]}\n'
        f'License: {package["license"] or "See license files"}\n'
        f'Source: {package["source"]}\n'
    )
    candidates = set()
    for path in root.rglob("*"):
        name = path.name.lower()
        if path.is_file() and (
            name.startswith(("license", "licence", "copying", "copyright", "notice"))
            or name == "readme.ijg"
        ) and path.suffix not in {".rs", ".py", ".c", ".h"}:
            candidates.add(path)
    if package["license_file"]:
        candidates.add(root / package["license_file"])
    for path in sorted(candidates):
        output = target / path.relative_to(root)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, output)

sysroot = Path(subprocess.check_output(["rustc", "--print", "sysroot"], text=True).strip())
for name in ("COPYRIGHT.html", "LICENSE-APACHE", "LICENSE-MIT"):
    path = sysroot / "share/doc/rust" / name
    if path.is_file():
        shutil.copyfile(path, destination / f"rust-{name}")
