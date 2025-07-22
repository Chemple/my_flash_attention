#include "utils.cuh"
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <__clang_cuda_builtin_vars.h>
#include <cuda.h>
#include <sys/cdefs.h>

#define OFFSET(col_num, row_idx, col_idx) ((row_idx) * (col_num) + (col_idx))

template <typename data_type = float, uint32_t seq_len, uint32_t dim,
          uint32_t Bc, uint32_t Br, data_type NEG_INF>
void flash_attention_impl(data_type const* d_Q, data_type const* d_K,
                          data_type const* d_V, data_type* d_O) {
  // init
  auto h_vec = std::vector<data_type>(0, seq_len * dim);
  CUDA_CHECK(cudaMemcpy(d_O, h_vec.data(), sizeof(data_type) * dim * seq_len,
                        cudaMemcpyHostToDevice));

  dim3 block_size{Bc, Br};
  dim3 grid_size{seq_len / Br};
  flash_attention_impl<data_type, seq_len, dim, Bc, Br, NEG_INF>(d_Q, d_K, d_V,
                                                                 d_O);
  CUDA_CHECK(cudaGetLastError());
}

template <typename data_type = float, uint32_t seq_len, uint32_t dim,
          uint32_t Bc, uint32_t Br, data_type NEG_INF>
void __global__ flash_attn_fwd_kernel_wo_mask(const data_type __restrict__* Q,
                                              const data_type __restrict__* K,
                                              const data_type __restrict__* V,
                                              data_type __restrict__* O,
                                              data_type scale_factor) {
  uint32_t block_idx = blockIdx.x;
  // TODO(shiwen): check this to ensure collesced memory access
  uint32_t thread_row_idx = threadIdx.y;
  uint32_t thread_col_idx = threadIdx.x;

  __shared__ data_type q_block[Br][dim];
  __shared__ data_type k_block[Bc][dim];
  __shared__ data_type v_block[Bc][dim];
  __shared__ data_type o_block[Br][dim];

  // data_type reg_x = 0;
  // data_type reg_m = min;
  // data_type reg_l = 0;

  __shared__ data_type sim[Br][Bc];
  __shared__ data_type m[Br];
  __shared__ data_type l[Br];
  // data_type thread_sim = 0;
  // data_type iter_m = NEG_INF;
  // data_type thread_l = 0;

  // load Q into shared memoy.
  for (uint32_t j = thread_col_idx; j < dim; j += Bc) {
    q_block[thread_row_idx][j] =
        Q[OFFSET(dim, block_idx * Br + thread_row_idx, j)];
    // init o_block
    o_block[thread_row_idx][j] = 0;
  }

  // init m and l
  m[thread_row_idx] = NEG_INF;
  l[thread_row_idx] = 0;

  constexpr uint32_t kv_block_num = seq_len / Bc;
  for (uint32_t kb_block_idx = 0; kb_block_idx < kv_block_num; kb_block_idx++) {
    // load kv block from global memory to shared memory.
    for (uint32_t i = thread_row_idx; i < Bc; i += Br) {
      for (uint32_t j = thread_col_idx; j < dim; j += Bc) {
        k_block[i][j] = K[OFFSET(dim, kb_block_idx * Bc + i, j)];
        v_block[i][j] = V[OFFSET(dim, kb_block_idx * Bc + i, j)];
      }
    }

    __syncthreads();

    // calculate q_block * k_block^T
    // TODO(shiwen): use thread_tiling to gain more performance
    data_type thread_sim = 0;
    for (uint32_t k = 0; k < dim; k++) {
      thread_sim += q_block[thread_row_idx][k] * k_block[thread_col_idx][k];
    }
    thread_sim *= scale_factor;
    sim[thread_row_idx][thread_col_idx] = thread_sim;

    __syncthreads();

    // TODO(shiwen): use reduce compute to gain more performance. ⬇️
    // TODO(shiwen): if Bc is ngt warp_size, do not use shared memory. use
    // register warp-level primitives ⬇️

    // TODO(shiwen): naive impl, three pass, but can be reduced to one pass
    data_type block_m = NEG_INF;
    for (uint32_t i = 0; i < Bc; i++) {
      block_m = max(sim[thread_row_idx][thread_col_idx], block_m);
    }
    data_type new_max = max(m[thread_row_idx], block_m);
    // TODO(shiwen): need this sync?
    __syncthreads();

    sim[thread_row_idx][thread_col_idx] =
        exp(sim[thread_row_idx][thread_col_idx] - new_max);
    __syncthreads();

    data_type block_l = 0;
    for (int i = 0; i < Bc; i++) {
      block_l += sim[thread_row_idx][i];
    }
    __syncthreads();

    data_type recover_factor = exp(m[thread_row_idx] - new_max);

    // recover l[thread_row_idx] and update new_l
    data_type new_l = block_l + l[thread_row_idx] * recover_factor;

    // recover o
    for (uint32_t i = thread_row_idx; i < Bc; i += Br) {
      for (uint32_t j = thread_col_idx; j < dim; j += Bc) {
        o_block[i][j] = o_block[i][j] * recover_factor;
      }
    }
    __syncthreads();

    for (uint32_t d = 0; d < dim; d++) {
      o_block[thread_col_idx][d] +=
          sim[thread_row_idx][thread_col_idx] * v_block[thread_col_idx][d];
    }

    m[thread_row_idx] = new_max;
    l[thread_row_idx] = new_l;

    __syncthreads();
  }

  for (uint32_t i = thread_row_idx; i < Bc; i += Br) {
    for (uint32_t j = thread_col_idx; j < dim; j += Bc) {
      O[OFFSET(dim, block_idx * Bc + i, j)] = o_block[i][j] / l[thread_row_idx];
    }
  }
}
