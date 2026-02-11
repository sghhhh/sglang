#pragma once
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/vec.cuh>

namespace device::typetraits {

// ==================== VecTypeTrait ====================
// Maps a scalar type T + desired byte width → packed vector type + AlignedVector type.
// Shared by fused_add_rmsnorm and qknorm_across_heads kernels.

template <typename T, int VEC_SIZE_IN_BYTE>
struct VecTypeTrait;

template <>
struct VecTypeTrait<bf16_t, 16> {
  using packed_t = packed_t<bf16_t>;
  using vec_t = device::AlignedVector<packed_t, 4>;
};

template <>
struct VecTypeTrait<fp16_t, 16> {
  using packed_t = packed_t<fp16_t>;
  using vec_t = device::AlignedVector<packed_t, 4>;
};

template <>
struct VecTypeTrait<bf16_t, 32> {
  using packed_t = packed_t<bf16_t>;
  using vec_t = device::AlignedVector<packed_t, 8>;
};

template <>
struct VecTypeTrait<fp16_t, 32> {
  using packed_t = packed_t<fp16_t>;
  using vec_t = device::AlignedVector<packed_t, 8>;
};



}//device::typetraits 