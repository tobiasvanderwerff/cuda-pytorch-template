"""Simple script for benchmarking CUDA kernels."""

import time
import argparse
import torch

# Note: We need to import the CUDA kernels *after* importing torch
import my_cuda_kernels


def benchmark(f, iters, *args):
    """
    Note: This is a simple benchmarking function that does not take into account
    the overhead of the Python interpreter and CUDA device initialization. For
    more accurate results, consider using the PyTorch profiler or Nvidia ncu.
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        out = f(*args)
    torch.cuda.synchronize()
    t1 = time.perf_counter_ns()
    print(f"Avg. time: {(t1-t0)/iters/1000:.2f} µs")

    """
    Uncomment the following block to run the PyTorch profiler, which provides
    more detailed performance metrics.
    """
    # print("\nRunning PyTorch profiler...")
    # with torch.profiler.profile() as prof:
    #     for i in range(iters):
    #         out = f(*args)
    #         torch.cuda.synchronize()
    # print(prof.key_averages().table())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark CUDA kernels.')
    parser.add_argument('-i', '--iters', type=int, default=1000, help='Number of iterations for benchmarking')
    args = parser.parse_args()

    n = 1000

    torch.manual_seed(1)

    m1 = torch.randn(n, n, device="cuda")
    m2 = torch.randn(n, n, device="cuda")

    print(f"\nBenchmarking matmul kernel on input size ({n}, {n})")

    benchmark(my_cuda_kernels.matmul, args.iters, m1, m2)