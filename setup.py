"""Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/setup.py"""

import warnings
import os
from pathlib import Path
from packaging.version import parse

from setuptools import setup, find_packages
import subprocess

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
)

PACKAGE_NAME = "cuda_template"  # name of the Python package
PACKAGE_IMPORT_NAME = "my_cuda_kernels"  # the name that you will import in Python

# Select your GPUs compute capability for faster compilation
COMPUTE_CAPABILITY = None  
# COMPUTE_CAPABILITY = "75"  # Turing 
# COMPUTE_CAPABILITY = "80"  # Ampere


# os.environ['CXX'] = '/usr/lib/ccache/g++'
# os.environ['CC'] = '/usr/lib/ccache/gcc'


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    return nvcc_extra_args + ["--threads", nvcc_threads]


print("\nTorch version = {}".format(torch.__version__))

if CUDA_HOME is not None:
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    print(f"CUDA version = {bare_metal_version}")
else:
    warnings.warn("CUDA_HOME is not set.")
    
cc_flag = []
if COMPUTE_CAPABILITY is not None:
    cc_flag.append("-gencode")
    cc_flag.append(f"arch=compute_{COMPUTE_CAPABILITY},code=sm_{COMPUTE_CAPABILITY}")

suffixes = [".cpp", ".cu"]
sources = [p for p in Path("csrc").rglob("*") if p.suffix in suffixes]

print(f"\nFound sources: {[str(p) for p in sources]}\n\n")

ext_modules = [
    CUDAExtension(
        name=PACKAGE_IMPORT_NAME,
        sources=sources,
        extra_compile_args={
            "cxx": ["-O2"],
            "nvcc": append_nvcc_threads(["-O2"] + cc_flag),
        },
        include_dirs=[],
    )
]

class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            import psutil

            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, os.cpu_count() // 2)

            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)  # free memory in GB
            max_num_jobs_memory = int(free_memory_gb / 9)  # each JOB peak memory cost is ~8-9GB when threads = 4

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)

setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "tests",
            "dist",
        )
    ),
    ext_modules=ext_modules,
    cmdclass={"build_ext": NinjaBuildExtension},
    python_requires=">=3.8",
    install_requires=[
        "torch",
    ],
    setup_requires=[
        "packaging",
        "psutil",
        "ninja",
    ],
)