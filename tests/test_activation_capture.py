"""CPU contract checks for fixed-storage selected-token observations."""
from types import SimpleNamespace
import unittest
import torch
from ignition.components.activation_capture import ActivationCapture

class TokenCaptureTests(unittest.TestCase):
    def test_selected_prefill_padding_and_shared_decode_storage(self):
        capture=ActivationCapture.allocate(2,4,3,'cpu',intermediate=6)
        capture.token_indices[1,:2]=torch.tensor([2,5])
        capture.token_enabled[1,:2]=True
        capture.begin(SimpleNamespace(positions=torch.tensor([0,1,-1,-1]),live_counts=torch.tensor([2])))
        values=torch.arange(24,dtype=torch.float32).reshape(4,6)
        capture.record_mlp_input(values,1)
        self.assertTrue(torch.equal(capture.token_values[1,:2,:2],values[:2][:,[2,5]]))
        self.assertFalse(bool(capture.token_values[1,2:].any()))
        self.assertFalse(bool(capture.token_values[1,:,2:].any()))
        self.assertTrue(torch.equal(capture.mlp_input_last[1],values[1]))
        decode=capture.rows(1)
        self.assertEqual(decode.token_indices.data_ptr(),capture.token_indices.data_ptr())
        decode.begin(SimpleNamespace(positions=torch.tensor([2]),live_counts=torch.tensor([1])))
        decode.record_mlp_input(values[:1]+10,1)
        self.assertTrue(torch.equal(capture.token_values[1,0,:2],values[0,[2,5]]+10))

    def test_inactive_capture_cannot_expose_stale_rows(self):
        capture=ActivationCapture.allocate(1,2,2,'cpu',intermediate=3)
        capture.token_enabled.fill_(True)
        capture.begin(SimpleNamespace(positions=torch.tensor([-1,-1]),live_counts=torch.tensor([0])))
        values=torch.ones((2,3))
        capture.record_mlp_input(values,0)
        capture.record_residual(torch.ones((2,2)),0)
        self.assertFalse(bool(capture.token_values.any()))
        self.assertFalse(bool(capture.mlp_input_last.any()))
        self.assertFalse(bool(capture.residual_l2.any()))

if __name__=='__main__':unittest.main()
