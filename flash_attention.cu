#include "utils.cuh"
#include <cstdint>
#include <cuda.h>
#include <vector>

template <typename data_type = float, uint32_t seq_len, uint32_t head_dim,
          uint32_t Bc, uint32_t Br, uint32_t Tc, uint32_t Tr>
void flash_attention_impl(const data_type *d_Q, const data_type *d_K,
                          const data_type *d_V, data_type *d_O) {
  // init
  auto h_vec = std::vector<data_type>(0, seq_len * head_dim);
  CUDA_CHECK(cudaMemcpy(d_O, h_vec, sizeof(data_type) * head_dim * seq_len,
                        cudaMemcpyHostToDevice));

  dim3 block_size{Bc / Tc,Tr};
  dim3 grid_size{};
}