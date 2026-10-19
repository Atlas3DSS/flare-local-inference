"""Qualify bounded remote b12x transfers on an explicitly assigned CUDA device."""

import json
import os
import socket
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from safetensors.torch import save_file

from ignition.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.checkpoint_source import RootSource, RemoteSource, SourceServer


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


@torch.no_grad()
def test_remote_native_copies_preserve_bits_strides_expansion_and_csf(tmp_path):
    root = tmp_path / "synthetic-source"
    root.mkdir()
    fixed = torch.tensor(([124] * 16 + [0] * 16) * 2, dtype=torch.uint8)
    tensors = {"packed": torch.arange(128, dtype=torch.uint8).reshape(8, 16),
               "bf16": torch.arange(32, dtype=torch.bfloat16).reshape(4, 8),
               "wide": torch.arange(80, dtype=torch.uint8).reshape(2, 40),
               "calibration.weight_scale_2": torch.tensor([.75]),
               "empty": torch.empty(0, dtype=torch.uint8),
               "scale.mxfp4_csf_fixed": fixed,
               "scale.mxfp4_csf_exceptions": torch.tensor([127 << 24], dtype=torch.uint32),
               "engram": torch.ones(1 << 20, dtype=torch.uint8)}
    save_file(tensors, str(root / "weights.safetensors"))
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(value.nbytes for value in tensors.values())},
        "weight_map": {name: "weights.safetensors" for name in tensors}}))
    (root / "config.json").write_text('{}')
    checkpoint = SimpleNamespace(repository=str(root), revision="synthetic")
    source = RootSource(root, checkpoint)
    sender, receiver = socket.socketpair()
    sender.settimeout(30)
    receiver.settimeout(30)
    server = SourceServer(source, sender, b"fixture-key" * 4, 1, "synthetic-code")
    server.thread.start()
    remote = None
    class Loader(DirectWeightLoader):
        def _selected_names(self):
            return super()._selected_names() - {"engram"}
    try:
        remote = RemoteSource(receiver, b"fixture-key" * 4, 1, "synthetic-code", checkpoint, directory=tmp_path)
        server.wait_ready()
        with remote.activate():
            manifest = inspect_indexed_checkpoint(remote.snapshot)
            with Loader(manifest, "cuda:0") as loader, \
                    mock.patch("ignition.preparation.remote_weights.BLOCK_BYTES", 32):
                packed = loader.load_tensor("packed", extent=(slice(None), slice(3, 9)))
                torch.testing.assert_close(packed.cpu(), tensors["packed"][:, 3:9], rtol=0, atol=0)
                expanded = loader.load_tensor("bf16", dtype=torch.float32, extent=(slice(None), slice(1, 4)))
                torch.testing.assert_close(expanded.cpu(), tensors["bf16"][:, 1:4].float(), rtol=0, atol=0)
                wide = loader.load_tensor("wide")
                torch.testing.assert_close(wide.cpu(), tensors["wide"], rtol=0, atol=0)
                scalar = loader.load_tensor("calibration.weight_scale_2")
                assert scalar.item() == .75
                assert loader.load_tensor("empty").numel() == 0
                strided = torch.full((8, 12), 231, dtype=torch.uint8, device="cuda:0")
                loader._copy(strided[:, ::2], loader._sources["packed"][:, 3:9])
                loader._session.flush()
                torch.testing.assert_close(strided[:, ::2].cpu(), tensors["packed"][:, 3:9], rtol=0, atol=0)
                assert bool((strided[:, 1::2] == 231).all())
                interleaved = torch.full((8, 16), 231, dtype=torch.uint8, device="cuda:0")
                loader._copy(interleaved[:, :2], loader._sources["packed"][:, :2])
                loader._copy(interleaved[:, 2:4], loader._sources["packed"][:, 2:4])
                loader._session.flush()
                torch.testing.assert_close(interleaved[:, :4].cpu(), tensors["packed"][:, :4], rtol=0, atol=0)
                assert bool((interleaved[:, 4:] == 231).all())
                csf = loader._queue_csf_sources(["scale"], rows=32, columns=4, quantization="mxfp4")
                loader._session.flush()
                assert torch.equal(csf.components[0].cpu().reshape(-1), fixed)
                assert torch.equal(csf.components[1].cpu(), tensors["scale.mxfp4_csf_exceptions"])
                assert loader._session.torch_copy_bytes == 0
                assert not list(remote.snapshot.rglob("*.safetensors"))
            torch.cuda.synchronize()
            assert remote.stats["payload_bytes"] == 304
            remote.finish()
        server.join()
        assert server.stats == remote.stats
        print(json.dumps(dict(remote_source=remote.stats, gpu=torch.cuda.get_device_name(0)), sort_keys=True))
    finally:
        receiver.close()
        try:
            sender.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        server.thread.join(30)
        sender.close()
        if remote is not None:
            remote.close()
        source.close()
