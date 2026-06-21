#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

/**
 * repack_4_to_8_kernel
 *
 * Converts packed INT4 from 4-per-int32 (gappy, bits 0-15 used, 16-31 zero)
 * to 8-per-int32 (dense, all 32 bits used) for MMA consumption.
 *
 * src: [rows, K/4] int32, each word = [s4_0|s4_1<<4|s4_2<<8|s4_3<<12]
 * dst: [rows, K/8] int32, each word = [s4_0|s4_1<<4|...|s4_7<<28]
 *
 * Transformation per output word:
 *   lo = src[row, i*2 + 0]   (4 s4 values in bits 0-15)
 *   hi = src[row, i*2 + 1]   (4 s4 values in bits 0-15)
 *   dst[row, i] = lo | (hi << 16)
 */
__global__ void repack_4_to_8_kernel(
    const int32_t* __restrict__ src,
    int32_t* __restrict__ dst,
    int rows,
    int K
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int src_cols = K / 4;
    int dst_cols = K / 8;

    for (int i = tid; i < dst_cols; i += blockDim.x) {
        int32_t lo = src[row * src_cols + i * 2];
        int32_t hi = src[row * src_cols + i * 2 + 1];
        dst[row * dst_cols + i] = lo | (hi << 16);
    }
}

void launch_repack_4_to_8(
    const int32_t* src,
    int32_t* dst,
    int rows,
    int K,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = rows;
    repack_4_to_8_kernel<<<blocks, threads, 0, stream>>>(src, dst, rows, K);
}

/**
 * fused_int4_tc_per_token_kernel
 *
 * INT4 Tensor Core GEMM for SM 12.1 (Blackwell).
 * Uses mma.sync.aligned.m16n8k64.row.col.s4.s4.s32 via nvcuda::wmma.
 *
 * x_packed: [M, K/8] int32, dense 8 s4 per word (post repack_4_to_8)
 * w_packed: [N, K/8] int32, dense 8 s4 per word
 * w_scale:  [N] float32, per-output-channel weight scale
 * s_x:      [M] float32, per-token activation scale
 * y:        [M, N] bfloat16, output
 *
 * Grid:
 *   blockIdx.x * 64 in N dimension (8 warps × 8 cols each)
 *   blockIdx.y * 16 in M dimension
 *   256 threads/block (8 warps)
 *   Shared memory: 16×64×4 = 4096 bytes
 */
#include <mma.h>

namespace wmma = nvcuda::wmma;

__global__ void fused_int4_tc_per_token_kernel(
    const int32_t* __restrict__ x_packed,
    const int32_t* __restrict__ w_packed,
    const float* __restrict__ w_scale,
    const float* __restrict__ s_x,
    __nv_bfloat16* __restrict__ y,
    int M,
    int N,
    int K
) {
    const int warp_id = threadIdx.x / 32;
    const int num_warps = blockDim.x / 32;

    const int m_start = blockIdx.y * 16;
    const int n_start = blockIdx.x * 64;

    if (m_start >= M) return;

    int ldm = K / 2;

    const int8_t* x_base = reinterpret_cast<const int8_t*>(
        x_packed + m_start * (K / 8)
    );
    const int8_t* w_base = reinterpret_cast<const int8_t*>(
        w_packed + n_start * (K / 8)
    );

    wmma::Fragment<wmma::accumulator, 16, 8, 64, int32_t> acc;
    wmma::fill_fragment(acc, 0);

    for (int k_offset = 0; k_offset < K; k_offset += 64) {
        wmma::Fragment<wmma::matrix_a, 16, 8, 64, int8_t, wmma::row_major> a_frag;
        wmma::Fragment<wmma::matrix_b, 16, 8, 64, int8_t, wmma::col_major> b_frag;

        wmma::load_matrix_sync(a_frag, x_base + k_offset / 2, ldm);
        wmma::load_matrix_sync(b_frag, w_base + k_offset / 2, ldm);
        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }

    __shared__ float tile[16][64];

    wmma::store_matrix_sync(
        (float(*)[8])&tile[0][warp_id * 8],
        acc, 64, wmma::mem_row_major
    );
    __syncthreads();

    int tid = threadIdx.x;
    if (tid < 16 * 64) {
        int row = m_start + tid / 64;
        int col = n_start + tid % 64;
        if (row < M && col < N) {
            float val = tile[tid / 64][tid % 64] * s_x[row] * w_scale[col];
            y[row * N + col] = __float2bfloat16(val);
        }
    }
}

void launch_fused_int4_tc_per_token(
    const int32_t* x_packed,
    const int32_t* w_packed,
    const float* w_scale,
    const float* s_x,
    __nv_bfloat16* y,
    int M,
    int N,
    int K,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((N + 63) / 64, (M + 15) / 16);
    fused_int4_tc_per_token_kernel<<<grid, block, 0, stream>>>(
        x_packed, w_packed, w_scale, s_x, y, M, N, K
    );
}
