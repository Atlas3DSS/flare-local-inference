# Build Flare for SM70

This page builds the two halves of Flare for a Tesla V100:

- the native `flare` server, a Rust binary;
- the Python preparation environment (Ignition, PyTorch, Triton). It compiles
  the SM70 CUDA kernels when it prepares a package.

The upstream [environment setup](../model-preparation.md#environment-setup)
applies, with the SM70 differences below. Run every command from the repository
root.

## Toolchain

| Component | Requirement | Recorded in the fork's runs |
| --- | --- | --- |
| GPU | Tesla V100 32 GB, compute capability 7.0 | Tesla V100-PCIE-32GB (commit `a6712de`) |
| NVIDIA driver | A driver that supports CUDA 12.9. R580 is the last driver branch that supports Volta. | Fresh V100 validation: 580.178.04. The historical runs did not record the exact version. |
| CUDA toolkit (`nvcc`, CUPTI) | 12.x with `sm_70` support. CUDA 13 cannot compile for `sm_70`. | Fresh V100 validation: `nvcc` 12.9.86. Historical runs recorded CUDA 12.9 (`b5bea16`). |
| Python | 3.12 (`pyproject.toml`: `>=3.12,<3.13`) | 3.12.3 |
| PyTorch | 2.14.1 built for CUDA 12.6 (`+cu126`) | 2.14.1+cu126 |
| Triton | the version that PyTorch 2.14.1 installs | 3.8.0 |
| Rust | 1.88 or later (`rust-version` in `flare/Cargo.toml`) | rustc 1.97.1 (STATUS.md line 8) |
| `protoc` | Protobuf compiler, for the runtime-package schema | The first native build used the `protoc` bundled with `grpcio-tools` ([below](#build-the-native-server)); its version was not recorded. |
| CUTLASS headers | `include/cutlass/gemm/warp/mma_tensor_op_sm70.h` and `mma_tensor_op_policy.h` | Revision not recorded. These docs use tag `v4.8.1`, which contains both headers. |

`pyproject.toml` pins `torch==2.14.1+cu132`. CUDA 13 dropped Volta, so those
wheels do not run on a V100. Replace them with the `cu126` build as shown below.

## System packages

On Ubuntu 24.04:

```sh
sudo apt-get install -y build-essential cmake ninja-build nasm pkg-config protobuf-compiler libssl-dev python3-dev liburing-dev git curl
```

Install CUDA 12.9 from NVIDIA's repository for your distribution. It goes into
`/usr/local/cuda-12.9` by default. Install Rust with `rustup` and `uv` with its
installer if you do not have them:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Check the toolchain:

```sh
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
/usr/local/cuda-12.9/bin/nvcc --version
rustc --version
protoc --version
```

`compute_cap` must be `7.0`.

## Build the native server

```sh
cargo build --release --locked --manifest-path flare/Cargo.toml
./flare/target/release/flare --version
./flare/target/release/flare run --help
```

The binary is `flare/target/release/flare`. It needs no CUDA toolkit; it loads
the NVIDIA driver at run time. Building it needs no GPU, so you can build on
another Linux host and copy it.

To run the native unit tests, including the scheduler progress tests that guard
the hang fix ([ARCHITECTURE.md](ARCHITECTURE.md#1-scheduler-hang-fix)):

```sh
cargo test --locked --release --manifest-path helios/Cargo.toml --target-dir flare/target --features runtime-package,server --lib
cargo test --locked --release --manifest-path helios/Cargo.toml --target-dir flare/target --features runtime-package,server --lib progress_tests
cargo test --locked --release --manifest-path flare/Cargo.toml
```

The Helios tests need `protoc`. On `fe058dc` the progress tests report
`4 passed; 0 failed; 1 ignored` (the ignored one is an exploration that prints
replay outcomes).

Upstream CI runs the Helios command without `--lib`. On this tree that also
builds the integration test `helios/tests/deepseek_requests.rs`, which does not
compile (`no field images on type Prompt`). The fork does not touch that test or
the `Prompt` struct; the error is inherited from the upstream base `e7b8c01`.
Upstream CI also raises the locked-memory limit first
(`prlimit --memlock=67108864`); do the same if a test fails to register host
memory.

If you cannot install `protobuf-compiler`, `prost-build` accepts any `protoc`
named by the `PROTOC` environment variable. The fork's first native build used
the compiler bundled with the `grpcio-tools` Python package, through a
four-line wrapper:

```sh
python3 -m venv /data/src/protoc-venv && /data/src/protoc-venv/bin/pip install grpcio-tools
printf '#!/data/src/protoc-venv/bin/python\nimport sys\nfrom grpc_tools.protoc import main\nsys.exit(main(sys.argv))\n' > /data/src/protoc && chmod +x /data/src/protoc
export PROTOC=/data/src/protoc
```

## Build the Python environment

```sh
uv sync --locked --extra prepare --extra test
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu126 --reinstall-package torch --reinstall-package torchvision torch==2.14.1+cu126 torchvision==0.29.1+cu126
```

The first command installs the locked environment, including Ignition and the
`flare_heliopause` extension, which it builds with Cargo. The second replaces
PyTorch with the CUDA 12.6 build. If you run `uv sync` again later, it restores
the `cu132` wheels; repeat the second command afterwards. Call `.venv/bin/python`
directly rather than `uv run`, which syncs first.

The fork's first SM70 run recorded these package versions in its preparation
environment: `torch` 2.14.1+cu126, `triton` 3.8.0, `flashinfer-python` 0.7.1,
`transformers` 5.21.0.dev0, `nvidia-cutlass-dsl` 4.7.1, `cuda-python` 13.3.1,
`nvidia-cuda-cupti-cu12` 12.6.80, `safetensors` 0.8.0, `accelerate` 1.14.0,
`huggingface-hub` 1.33.0, `tokenizers` 0.23.2, `protobuf` 7.36.2. The release
runs used the same environment directory; its package list was not re-recorded
for them. Apart from PyTorch, `uv.lock` is the reference.

Use CUDA 12 CUPTI for V100 preparation and package tests. If the environment
also contains a CUDA 13 CUPTI wheel, automatic library discovery can select it
and fail with `CUPTI_ERROR_INVALID_DEVICE`. Preload the installed CUDA 12
library in the preparation/test shell before running Python:

```sh
FLARE_CUPTI12=$(.venv/bin/python -c 'import pathlib, nvidia.cuda_cupti; print(pathlib.Path(next(iter(nvidia.cuda_cupti.__path__))) / "lib/libcupti.so.12")')
test -f "$FLARE_CUPTI12"
export LD_PRELOAD="$FLARE_CUPTI12${LD_PRELOAD:+:$LD_PRELOAD}"
.venv/bin/python -c 'from cuda.pathfinder import load_nvidia_dynamic_lib; print(load_nvidia_dynamic_lib("cupti").abs_path)'
```

The last command must resolve `libcupti.so.12`. Native serving does not require
CUPTI; this setting is for the Python preparation environment.

Point the build at CUDA 12.9 and fetch CUTLASS:

```sh
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
git clone --depth 1 --branch v4.8.1 https://github.com/NVIDIA/cutlass.git /data/src/cutlass
```

The SM70 kernels compile with `$CUDA_HOME/bin/nvcc` (PyTorch's `CUDA_HOME`).
Only the prefill projection library includes CUTLASS: its warp-level SM70 MMA
headers and the headers they include. CUTLASS is BSD-3-Clause; the license is in
`ignition/components/CUTLASS-LICENSE.txt`.

Check that PyTorch sees the card as `sm_70`:

```sh
.venv/bin/python -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())"
```

Expect `2.14.1+cu126`, `12.6`, `(7, 0)`, and `sm_70` in the architecture list.

## How the SM70 kernels are built

There is no separate kernel build step. During preparation each SM70 component
compiles its CUDA source with `nvcc -arch=compute_70 -code=sm_70`. The
projection, prefill, attention and GDN prefill builds add `--fmad=false`; the
dense and commit builds do not. The prefill build also adds `-DSOL_STAGE=1` and
the CUTLASS include directory. Each library is
cached as `<sha256>.so` in the kernel cache you pass to preparation. The key
covers the CUDA source, the Python wrapper and, for prefill, every CUTLASS
header it reads, so a changed header compiles a new library. The Gated DeltaNet
kernels are Triton kernels, compiled for `sm_70` by Triton with floating-point
fusion disabled.

Libraries must be compiled before CUDA graph capture. A library that is not
already loaded fails with "must be loaded before capture" instead of compiling
during capture.

## Confirm the SM70 build

**1. Build paths and the reviewed kernel (CPU, no GPU needed).**
`tests/components/test_sm70_build.py` loads the build functions with `nvcc`
replaced. It checks that every SM70 library builds with `$CUDA_HOME/bin/nvcc`
into the configured kernel cache, and that the prefill build requires an
explicit CUTLASS directory whose transitive headers change the cache key. It
also checks that every served pgemm tile is a compiled instantiation inside the
reviewed template space, and pins `sm70_pgemm.cuh` at SHA-256 `7d979e47...`:
the reviewed kernel (`60a4ed9f...`) plus the output row stride `ldy` of the
prefill fusions
([ARCHITECTURE.md](ARCHITECTURE.md#5-512-token-prefill-exact-fusions-and-the-g128-split-plan)).

```sh
.venv/bin/python -m pytest -q tests/components/test_sm70_build.py
```

On `fe058dc` this passes 4 tests (`python3 -m unittest` also runs the file
without pytest: "Ran 4 tests ... OK").

This test does not run `nvcc`. To prove the kernels compile and run, use step 2.

**2. Compile and run on the V100.** The CUDA component tests compile the SM70
libraries with the real `nvcc` and run them on the card. They refuse to run
unless the V100 is assigned explicitly with `CUDA_VISIBLE_DEVICES`:

```sh
CUDA_VISIBLE_DEVICES=0 FLARE_RUN_CUDA_TESTS=1 .venv/bin/python -m pytest -q -rA tests/components/test_sm70_projection_cuda.py tests/components/test_sm70_attention_cuda.py tests/components/test_sm70_decode_attention_cuda.py tests/components/test_sm70_prefill_attention_cuda.py tests/components/test_sm70_gdn_cuda.py tests/components/test_sm70_gdn_prefill_cuda.py tests/components/test_sm70_gdn_replay.py tests/components/test_sm70_embedding_cuda.py tests/components/test_sm70_sampling_cuda.py
```

Without `FLARE_RUN_CUDA_TESTS=1`, the CUDA tests are skipped, not failed. A
skipped CUDA test proves nothing. No CUDA test builds the prefill projection
library, which needs CUTLASS; preparation compiles it first. [VALIDATE.md](VALIDATE.md#gate-1-unit-and-regression-tests)
lists the full test groups and gate commands. The public runner at `b4987de5`
passed both GPU groups with no failures, exemptions or skips
([tests-A.log](receipts/tests-A.log), [tests-B.log](receipts/tests-B.log)); see the
[current validation record](ARCHITECTURE.md#public-release-review-and-current-validation).

**3. Inspect a compiled library.** After a test run or a preparation, every
library in the kernel cache should contain `sm_70` code only:

```sh
for f in /data/flare/kernel-cache/*.so; do echo "$f"; /usr/local/cuda-12.9/bin/cuobjdump --list-elf "$f"; done
```

## Common failures

| Symptom | Cause | Fix |
| --- | --- | --- |
| `no kernel image is available for execution on the device`, or `sm_70` missing from `torch.cuda.get_arch_list()` | The `cu132` PyTorch wheels from `uv sync` | Rerun the `uv pip install ... cu126` command |
| `nvcc fatal: Unsupported gpu architecture 'compute_70'` | `CUDA_HOME` or `PATH` points at CUDA 13 | `export CUDA_HOME=/usr/local/cuda-12.9` |
| `... preparation requires the configured CUDA toolkit` | PyTorch found no `CUDA_HOME` | Set `CUDA_HOME` before starting Python |
| `SM70 prefill requires a configured cutlass_include directory` | No `--cutlass-include` was passed | Pass the CUTLASS `include` directory |
| `SM70 prefill CUTLASS header is missing: ...` | The path is not a CUTLASS `include` directory | Point at `<cutlass>/include`, which contains `cutlass/gemm/warp/` |
| Triton fails to assemble for `sm_70` with a `ptxas` error | Triton's own `ptxas` cannot target Volta | `export TRITON_PTXAS_PATH=$CUDA_HOME/bin/ptxas` |
| `qualification requires sm_70`, or the preparation script refuses the GPU | GPU 0 in this process is not a V100 | Select the V100 with `CUDA_VISIBLE_DEVICES` |
| `could not find protoc` or a `prost-build` error during `cargo build`, `cargo test` or `uv sync` | `protoc` is not installed | `sudo apt-get install protobuf-compiler` |
| `uv sync` fails building `flare-heliopause` | Rust or the system packages are missing | Install them as above, then rerun |
| CUDA tests all `SKIPPED` | `FLARE_RUN_CUDA_TESTS` is unset | Set `FLARE_RUN_CUDA_TESTS=1` |
| `test_recorded_preparation_preserves_live_pages_and_full_bf16_range` reports no benchmarked candidates | A warm, shared b12x selection cache (STATUS.md line 39) | Give each test run a fresh `B12X_COMPILE_CACHE_DIR` |
| `CUDA out of memory` while compiling | Too many compiler workers for the card | Lower `--compile-workers`, or free the GPU |

The fork's Windows build was not attempted for SM70.
