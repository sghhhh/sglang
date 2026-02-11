#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/impl/cooperative_norm.cuh>

#include <tvm/ffi/container/tensor.h>

namespace {

using namespace device::cooperative_norm;

template <typename T, int VEC_SIZE_IN_BYTE>
__global__ void qknorm_across_heads_reg_kernel(
    T* __restrict__ q,
    T* __restrict__ k,
    const T* __restrict__ q_weight,
    const T* __restrict__ k_weight,
    int vec_hidden_size,
    float eps) {
  constexpr int inner_loop = kInnerLoop<VEC_SIZE_IN_BYTE>;

  __shared__ float shared_memory[64];  // 2 channels × 32

  using vec_t = typename VecTypeTrait<T, VEC_SIZE_IN_BYTE>::vec_t;
  using packed_t = typename VecTypeTrait<T, VEC_SIZE_IN_BYTE>::packed_t;
  vec_t v_q;         // Save q
  vec_t v_k;         // Save k
  vec_t v_q_weight;  // Save q_weight
  vec_t v_k_weight;  // Save k_weight
  vec_t v_q_out;     // Save q output
  vec_t v_k_out;     // Save k output

  auto token_id = blockIdx.x;
  float acc_square_q = 0.0f;
  float acc_square_k = 0.0f;

  if (threadIdx.x < vec_hidden_size) {
    // Compute address for q and k
    vec_t* p_q = reinterpret_cast<vec_t*>(q) + token_id * vec_hidden_size;
    vec_t* p_k = reinterpret_cast<vec_t*>(k) + token_id * vec_hidden_size;
    const vec_t* p_q_weight = reinterpret_cast<const vec_t*>(q_weight);
    const vec_t* p_k_weight = reinterpret_cast<const vec_t*>(k_weight);

    // Load data
    v_q = p_q[threadIdx.x];
    v_k = p_k[threadIdx.x];
    v_q_weight = p_q_weight[threadIdx.x];
    v_k_weight = p_k_weight[threadIdx.x];

    // Compute sum of squares for q and k
    acc_square_q = sum_of_squares<packed_t, VEC_SIZE_IN_BYTE>(v_q);
    acc_square_k = sum_of_squares<packed_t, VEC_SIZE_IN_BYTE>(v_k);
  }

  // CTA reduce → rsqrt (dual channel: Q and K simultaneously)
  int dim = vec_hidden_size * (VEC_SIZE_IN_BYTE / sizeof(T));
  device::cta::reduce_norm(shared_memory, eps, dim, acc_square_q, acc_square_k);
  // acc_square_q now contains rsqrt for Q, acc_square_k for K

  // Apply normalization and write back
  if (threadIdx.x < vec_hidden_size) {
    // Apply RMSNorm for Q
    for (int i = 0; i < inner_loop; i++) {
      v_q_out[i] = apply_rms_weight(v_q[i], v_q_weight[i], acc_square_q);
    }
    vec_t* p_q_out = reinterpret_cast<vec_t*>(q) + token_id * vec_hidden_size;
    p_q_out[threadIdx.x] = v_q_out;

    // Apply RMSNorm for K
    for (int i = 0; i < inner_loop; i++) {
      v_k_out[i] = apply_rms_weight(v_k[i], v_k_weight[i], acc_square_k);
    }
    vec_t* p_k_out = reinterpret_cast<vec_t*>(k) + token_id * vec_hidden_size;
    p_k_out[threadIdx.x] = v_k_out;
  }
}

template <typename DType>
struct QKNormAcrossHeadsKernel {
  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N, D})  // q
        .with_strides({D, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(q);
    TensorMatcher({N, D})  // k
        .with_strides({D, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(k);
    TensorMatcher({D})  // q_weight
        .with_dtype<DType>()
        .with_device(device)
        .verify(q_weight);
    TensorMatcher({D})  // k_weight
        .with_dtype<DType>()
        .with_device(device)
        .verify(k_weight);

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

      // Launch single kernel for both q and k
      auto kernel = max_vec_size_byte == 32 ? qknorm_across_heads_reg_kernel<DType, 32>
                                            : qknorm_across_heads_reg_kernel<DType, 16>;

      LaunchKernel(static_cast<uint>(N.unwrap()), threads, device.unwrap())
          .enable_pdl(false)(
              kernel,
              reinterpret_cast<DType*>(q.data_ptr()),
              reinterpret_cast<DType*>(k.data_ptr()),
              reinterpret_cast<DType*>(q_weight.data_ptr()),
              reinterpret_cast<DType*>(k_weight.data_ptr()),
              vec_hidden_size,
              eps);
    } else {
      host::RuntimeCheck(false, "Large hidden_sizes are not supported for now.");
    }
  }
};

}  // namespace
