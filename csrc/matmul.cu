/* Matrix multiplication kernel. */

#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>


__global__ void matmul_kernel(float* out, const float* A, const float* B, int h, int w, int k) {
    /* Naive implementation of matrix multiplication. */

    const int c = blockDim.x * blockIdx.x + threadIdx.x;
    const int r = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (r >= h || c >= w) return;

    float sum = 0.0f;
    for (int i = 0; i < k; ++i) sum += A[r*k + i] * B[i*w + c];
    out[r*w + c] = sum;
}

void launch_matmul_kernel(float* out, const float* A, const float* B, int h, int w, int k, dim3 grid_size, dim3 block_size) { 
    matmul_kernel<<<grid_size, block_size>>>(out, A, B, h, w, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}