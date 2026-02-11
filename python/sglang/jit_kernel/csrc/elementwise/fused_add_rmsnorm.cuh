#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/impl/cooperative_norm.cuh>

#include <tvm/ffi/container/tensor.h>

namespace {

using namespace device::cooperative_norm;

template <typename T, int VEC_SIZE_IN_BYTE>
__global__ void fused_add_rmsnorm_reg_kernel(
    T* __restrict__ input, T* __restrict__ residual, const T* __restrict__ weight, int vec_hidden_size, float eps) {
  constexpr int inner_loop = kInnerLoop<VEC_SIZE_IN_BYTE>;

  __shared__ float shared_memory[32];  // 1 channel × 32

  using vec_t = typename VecTypeTrait<T, VEC_SIZE_IN_BYTE>::vec_t;
  using packed_t = typename VecTypeTrait<T, VEC_SIZE_IN_BYTE>::packed_t;
  vec_t v;         // Save input
  vec_t v_res;     // Save residual
  vec_t v_weight;  // Save weight
  vec_t v_out;     // Save output

  auto token_id = blockIdx.x;
  float acc_square = 0.0f;

  if (threadIdx.x < vec_hidden_size) {
    // Compute address
    vec_t* p = reinterpret_cast<vec_t*>(input) + token_id * vec_hidden_size;
    vec_t* p_res = reinterpret_cast<vec_t*>(residual) + token_id * vec_hidden_size;
    const vec_t* p_weight = reinterpret_cast<const vec_t*>(weight);

    // Load data
    v = p[threadIdx.x];
    v_res = p_res[threadIdx.x];
    v_weight = p_weight[threadIdx.x];

    // Fused add + sum of squares
    for (int i = 0; i < inner_loop; i++) {
      float2 val = device::cast<fp32x2_t, packed_t>(v[i]);
      float2 res = device::cast<fp32x2_t, packed_t>(v_res[i]);
      float2 inp_res = make_float2(val.x + res.x, val.y + res.y);
      acc_square += inp_res.x * inp_res.x + inp_res.y * inp_res.y;
      v[i] = device::cast<packed_t, fp32x2_t>(inp_res);
    }

    // Store inp+res to residual
    p_res[threadIdx.x] = v;
  }

  // CTA reduce → rsqrt (single channel)
  int dim = vec_hidden_size * (VEC_SIZE_IN_BYTE / sizeof(T));
  device::cta::reduce_norm(shared_memory, eps, dim, acc_square);
  // acc_square now contains the rsqrt normalization factor

  // Compute RMSNorm
  if (threadIdx.x < vec_hidden_size) {
    for (int i = 0; i < inner_loop; i++) {
      v_out[i] = apply_rms_weight(v[i], v_weight[i], acc_square);
    }
    vec_t* p_out = reinterpret_cast<vec_t*>(input) + token_id * vec_hidden_size;
    p_out[threadIdx.x] = v_out;
  }
}

template <typename DType>
struct FusedAddRMSNormKernel {
  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView weight,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N, D})  // input
        .with_strides({D, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(input);
    TensorMatcher({D})  // weight
        .with_dtype<DType>()
        .with_device(device)
        .verify(weight);
    TensorMatcher({N, D})  // residual
        .with_strides({D, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(residual);

    auto cc_major = host::runtime::get_cc_major(device.unwrap().device_id);
    int hidden_size = static_cast<int>(D.unwrap());
    if (cooperative_norm::is_hidden_size_supported(cc_major, hidden_size)) {
      int max_vec_size_byte = cooperative_norm::get_max_vec_size_byte(cc_major);
      int elements_in_vec = cooperative_norm::get_elements_in_vec(max_vec_size_byte, sizeof(DType));
      int vec_hidden_size = cooperative_norm::get_vec_hidden_size(hidden_size, elements_in_vec);
      uint threads = cooperative_norm::get_threads(vec_hidden_size);

      // Runtime check
      host::RuntimeCheck(
          hidden_size % elements_in_vec == 0,
          "hidden_size",
          hidden_size,
          " can not align to elements_in_vec ",
          elements_in_vec);

      // Launch kernel
      auto kernel =
          max_vec_size_byte == 32 ? fused_add_rmsnorm_reg_kernel<DType, 32> : fused_add_rmsnorm_reg_kernel<DType, 16>;
      LaunchKernel(static_cast<uint>(N.unwrap()), threads, device.unwrap())
          .enable_pdl(false)(
              kernel,
              reinterpret_cast<DType*>(input.data_ptr()),
              reinterpret_cast<DType*>(residual.data_ptr()),
              reinterpret_cast<DType*>(weight.data_ptr()),
              vec_hidden_size,
              eps);
    } else {
      host::RuntimeCheck(false, "Large hidden_sizes are not supported for now.");
    }
  }
};

}  // namespace
