#pragma once
#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/type_traits.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/runtime.cuh>

// ==================== Device-side utilities ====================

namespace device::cooperative_norm {

// VecTypeTrait is defined in <sgl_kernel/type_traits.cuh> under device::typetraits
// Re-export for convenience
template <typename T, int VEC_SIZE_IN_BYTE>
using VecTypeTrait = typetraits::VecTypeTrait<T, VEC_SIZE_IN_BYTE>;

/**
 * \brief Number of packed elements per vector.
 * 16B → 4 packed pairs, 32B → 8 packed pairs.
 */
template <int VEC_SIZE_IN_BYTE>
inline constexpr int kInnerLoop = VEC_SIZE_IN_BYTE == 16 ? 4 : 8;

/**
 * \brief Apply RMS normalization weighting to a single packed element pair.
 * Computes: output = val * weight * rsqrt_square_sum
 */
template <typename PackedT>
SGL_DEVICE PackedT apply_rms_weight(PackedT& val, PackedT& weight, float rsqrt_square_sum) {
  float2 valf = cast<fp32x2_t, PackedT>(val);
  float2 weightf = cast<fp32x2_t, PackedT>(weight);
  return cast<PackedT, fp32x2_t>(
      make_float2(valf.x * weightf.x * rsqrt_square_sum, valf.y * weightf.y * rsqrt_square_sum));
}

/**
 * \brief Accumulate sum of squares from a vector of packed elements.
 * \return Sum of all x^2 values in the vector.
 */
template <typename PackedT, int VEC_SIZE_IN_BYTE, typename VecT>
SGL_DEVICE float sum_of_squares(const VecT& vec) {
  constexpr int inner_loop = kInnerLoop<VEC_SIZE_IN_BYTE>;
  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < inner_loop; i++) {
    float2 val = cast<fp32x2_t, PackedT>(vec[i]);
    acc += val.x * val.x + val.y * val.y;
  }
  return acc;
}

}  // namespace device::cooperative_norm

// ==================== Host-side utilities ====================

namespace host::cooperative_norm {

/**
 * \brief Max vector size in bytes based on compute capability.
 */
inline int get_max_vec_size_byte(int cc_major) {
  return cc_major >= 10 ? 32 : 16;
}

/**
 * \brief Max supported hidden size based on compute capability.
 */
inline int get_max_hidden_size(int cc_major) {
  return cc_major >= 10 ? 12288 : 8192;
}

/**
 * \brief Check if hidden size is supported for cooperative norm.
 */
inline bool is_hidden_size_supported(int cc_major, int hidden_size) {
  return hidden_size <= get_max_hidden_size(cc_major);
}

/**
 * \brief Number of elements per vector for a given dtype.
 */
inline int get_elements_in_vec(int max_vec_size_byte, int dtype_size) {
  return max_vec_size_byte / dtype_size;
}

/**
 * \brief Vectorized hidden size (hidden_size / elements_per_vec).
 */
inline int get_vec_hidden_size(int hidden_size, int elements_in_vec) {
  return hidden_size / elements_in_vec;
}

/**
 * \brief Thread block size, rounded up to warp size.
 */
inline uint32_t get_threads(int vec_hidden_size) {
  return static_cast<uint32_t>((vec_hidden_size + 31) / 32 * 32);
}

}  // namespace host::cooperative_norm
