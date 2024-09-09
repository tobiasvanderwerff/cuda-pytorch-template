/* Square kernel. */

#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>


__global__ void square_kernel(float *out, const float *inp, int n) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = inp[i] * inp[i]; 
} 

void launch_square_kernel(float* out, const float* inp, int n, int grid_size, int block_size) {
    square_kernel<<<grid_size, block_size>>>(out, inp, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}