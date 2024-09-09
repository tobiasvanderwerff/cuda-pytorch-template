import torch
import pytest

# Note: We need to import the CUDA kernels *after* importing torch
import my_cuda_kernels


ABS_TOL = 1e-4
REL_TOL = 1e-1


def test_square_kernel():
    torch.manual_seed(1)
    x = torch.randn(1000, device="cuda", dtype=torch.float32)

    out = my_cuda_kernels.square(x)
    out_pt = torch.square(x)

    assert torch.isclose(out, out_pt, atol=ABS_TOL, rtol=REL_TOL).all()
