#pragma once
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

namespace device::cta {

template <typename T>
SGL_DEVICE void reduce_max(T value, float* smem, float min_value = 0.0f) {
  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  smem[warp_id] = warp::reduce_max(value);
  __syncthreads();
  if (warp_id == 0) {
    const auto tx = threadIdx.x;
    const auto local_value = tx * kWarpThreads < blockDim.x ? smem[tx] : min_value;
    const auto max_value = warp::reduce_max(local_value);
    smem[0] = max_value;
  }
  // no extra sync; it is caller's responsibility to sync if needed
}

/**
 * \brief CTA-level reduction for RMSNorm: reduce sum-of-squares and compute rsqrt.
 *
 * Supports variadic channels (1 for fused_add_rmsnorm, 2 for qknorm_across_heads, etc.).
 * On entry, each `value` / `rest` contains a thread-local partial sum of squares.
 * On exit, each `value` / `rest` contains the rsqrt normalization factor (broadcast).
 *
 * Uses exactly 2 __syncthreads() calls regardless of channel count.
 * Requires smem of at least N * 32 floats (N = number of channels).
 *
 * \param smem Shared memory buffer
 * \param eps Epsilon for numerical stability
 * \param dim Total number of elements being normalized
 * \param value First channel's partial sum (modified in-place to rsqrt result)
 * \param rest Additional channels' partial sums (modified in-place to rsqrt results)
 */
template <typename T, typename... Ts>
SGL_DEVICE void reduce_norm(float* smem, float eps, int dim, T& value, Ts&... rest) {
  constexpr int N = 1 + sizeof...(Ts);
  const uint32_t warp_id = threadIdx.x / kWarpThreads;

  // Pack parameter pack into array for indexed access
  float arr[N] = {value, static_cast<float>(rest)...};

  // Step 0: Warp-level reduce each channel
#pragma unroll
  for (int i = 0; i < N; i++) {
    arr[i] = warp::reduce_sum(arr[i]);
  }

  // Write warp sums to smem (N channels, each using 32 floats)
  if (threadIdx.x % kWarpThreads == 0) {
#pragma unroll
    for (int i = 0; i < N; i++) {
      smem[i * 32 + warp_id] = arr[i];
    }
  }

  __syncthreads();

  // Step 1: First warp reduces across warps for each channel
  if (warp_id == 0) {
#pragma unroll
    for (int i = 0; i < N; i++) {
      float val = threadIdx.x < blockDim.x / kWarpThreads ? smem[i * 32 + threadIdx.x] : 0.0f;
      float cta_sum = warp::reduce_sum(val);
      // Write rsqrt to all 32 positions for broadcast
      smem[i * 32 + threadIdx.x] = math::rsqrt(eps + cta_sum / static_cast<float>(dim));
    }
  }

  __syncthreads();

  // Broadcast: each thread reads rsqrt from its warp's slot
#pragma unroll
  for (int i = 0; i < N; i++) {
    arr[i] = smem[i * 32 + warp_id];
  }

  // Unpack array back to parameter pack
  int idx = 0;
  value = arr[idx++];
  ((rest = arr[idx++]), ...);
}

}  // namespace device::cta
