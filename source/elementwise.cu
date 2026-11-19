// SPDX-License-Identifier: Apache-2.0
// add_kernel and silu_mul_kernel from pegainfer (provenance in NOTICE), with
// extern "C" linkage; the bodies are unchanged.
#include <cuda_bf16.h>

extern "C" __global__ void add_kernel(const __nv_bfloat16 *__restrict__ a, const __nv_bfloat16 *__restrict__ b,
                                      __nv_bfloat16 *__restrict__ out, int n) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x) {
    float va = __bfloat162float(a[idx]);
    float vb = __bfloat162float(b[idx]);
    out[idx] = __float2bfloat16(va + vb);
  }
}

extern "C" __global__ void silu_mul_kernel(const __nv_bfloat16 *__restrict__ gate, const __nv_bfloat16 *__restrict__ up,
                                           __nv_bfloat16 *__restrict__ out, int n) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x) {
    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    float silu_g = g / (1.0f + expf(-g));
    // Match Triton rounding: silu result cast to bf16 before multiply
    out[idx] = __float2bfloat16(__bfloat162float(__float2bfloat16(silu_g)) * u);
  }
}
