import torch
import pytest

# Note: We need to import the CUDA kernels *after* importing torch
import my_cuda_kernels


ABS_TOL = 1e-4
REL_TOL = 1e-1


@pytest.mark.parametrize(
    # We test on various input sizes 
    "h,k,w", [(2, 2, 20), (20, 2, 2), (3, 7, 11), (1024, 1024, 1024), 
              (1000, 10, 10), (10, 10, 1000), (999, 999, 999)]
)
def test_matmul_kernel(h, k, w):
    torch.manual_seed(1)
    m1 = torch.randn(h, k, device="cuda")
    m2 = torch.randn(k, w, device="cuda")

    out = my_cuda_kernels.matmul(m1, m2)
    out_pt = torch.matmul(m1, m2)

    assert torch.isclose(out, out_pt, atol=ABS_TOL, rtol=REL_TOL).all().item()
