# Native Windows serving

Flare targets Windows 11 x64 through Rust's `x86_64-pc-windows-msvc` target.
Ignition preparation remains on Linux. Copy the exported SM120 `.safetensors`
package and its publisher profile to the Windows serving host.

The Windows runtime uses the NVIDIA driver directly through `nvcuda.dll`.
Python, the CUDA toolkit, WSL, and Wine are not runtime dependencies.
Install the NVIDIA driver and the Microsoft Visual C++ x64 Redistributable.

## Build from Linux

Install Rust, Clang/LLVM with `lld-link`, CMake, Ninja, NASM, and Protobuf's
`protoc`, then run from the repository root:

```sh
rustup target add x86_64-pc-windows-msvc
cargo install --locked cargo-xwin --version 0.23.1
cargo xwin build --locked --release --manifest-path flare/Cargo.toml \
  --target x86_64-pc-windows-msvc
```

The executable is `flare/target/x86_64-pc-windows-msvc/release/flare.exe`.
`cargo-xwin` downloads the Microsoft SDK and CRT; its Microsoft license terms
apply. Use the release profile: the default cross-toolchain cache does not
include the debug CRT requested by libjpeg-turbo's debug configuration.

## Build and test on Windows

Install Rust with the MSVC toolchain, Visual Studio's C++ build tools and Windows
SDK, CMake, NASM, and `protoc`. Make NASM and `protoc` available on `PATH`.
From the repository root:

```powershell
cargo build --locked --release --manifest-path flare/Cargo.toml
cargo test --locked --release --manifest-path helios/Cargo.toml --features runtime-package,server
cargo test --locked --release --manifest-path flare/Cargo.toml
```

The `Native serving` GitHub Actions workflow builds on Linux and Windows,
runs the native CPU tests, and exercises both Windows executables on Windows.
GPU tests require explicitly supplied packages and hardware and remain ignored
in the CPU suite.

## Serve a prepared package

```powershell
.\flare\target\release\flare.exe run C:\models\qwen38-tp1-mtp5.safetensors `
  --publisher-keys C:\models\publisher.json --gpus 0 --bind 127.0.0.1:8001
```

Publisher acceptance, request sampling, graph-capacity selection, and HTTP APIs
follow the [CLI specification](cli-design.md). The default trust store is
`Flare\trust.toml` under the Windows Local App Data known folder. `--trust-store`
selects another path. Ctrl-C and Ctrl-Break drain native work before releasing
CUDA resources.

Disk embedding tables and disk prefix caches use unbuffered overlapped I/O with
bounded completion queues. File offsets, lengths, and buffers retain the package's
4 KiB alignment; incompatible storage geometry is rejected. RAM embedding tables
and memory prefix caches retain their existing explicit storage selections.
The open package handle prevents replacement or writes while it is loaded.

## Hardware compatibility

Native inference on GeForce GPUs under WDDM requires hardware validation.
CPU tests and successful linking do not establish CUDA compatibility.

The runtime preserves the package's recorded virtual addresses. Windows host
reservations are rounded to allocation granularity, with individual pages
committed at their recorded addresses. CUDA registration must preserve the
recorded GPU-visible host address. Address collisions and incompatible mappings
fail loading; the runtime does not relocate a package or change its graph contract.

Validate fixed device mappings, mapped host access, stream-memory waits, external
event ordering, and graph replay using the exported runtime fixtures before
running a model. Validate Qwen generation, ordinary/speculative transitions,
disk/RAM table delivery, prefix-cache restoration, cancellation, and shutdown
through the actual Windows service. Test tensor-parallel packages separately on
the intended GPU pair, including CUDA access to peer memory.

Multi-host RoCE serving, integrated GPUs, and native Windows preparation are not
supported. Windows uses the existing single-process, per-rank-thread execution
path for single-host tensor parallelism, subject to driver and device capabilities.
