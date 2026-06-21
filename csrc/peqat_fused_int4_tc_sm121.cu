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
 * INT4 Tensor Core GEMM for SM 12.1 (Blackwell) using inline PTX.
 * Instruction: mma.sync.aligned.m16n8k64.row.col.s4.s4.s32
 *
 * Fragment layout (per warp of 32 threads).
 * Formulas from PTX ISA v9.3 §9.7.14.5, Figures 93-96.
 *
 * A (row-major, 16x64 s4, 4 x uint32 per thread):
 *   groupID = lane >> 2, threadID_in_group = lane % 4
 *   a[0]: row = groupID,      col = threadID*8 .. threadID*8+7       (K first half)
 *   a[1]: row = groupID + 8,  col = threadID*8 .. threadID*8+7       (K first half)
 *   a[2]: row = groupID,      col = threadID*8+32 .. threadID*8+7+32 (K second half)
 *   a[3]: row = groupID + 8,  col = threadID*8+32 .. threadID*8+7+32 (K second half)
 *
 * B (col-major, 64x8 s4, 2 x uint32 per thread):
 *   groupID = lane >> 2, threadID_in_group = lane % 4
 *   b[0]: col = groupID,  row(=K) = threadID*8 .. threadID*8+7       (K first half)
 *   b[1]: col = groupID,  row(=K) = threadID*8+32 .. threadID*8+7+32 (K second half)
 *
 * D (accumulator, 16x8 int32, 4 x int32 per thread):
 *   groupID = lane >> 2, threadID_in_group = lane % 4
 *   d[0]: row = groupID,      col = threadID*2
 *   d[1]: row = groupID,      col = threadID*2 + 1
 *   d[2]: row = groupID + 8,  col = threadID*2
 *   d[3]: row = groupID + 8,  col = threadID*2 + 1
 *
 * x_packed: [M, K/8] int32, dense 8 s4 per word
 * w_packed: [N, K/8] int32, dense 8 s4 per word
 * w_scale:  [N] float32
 * s_x:      [M] float32
 * y:        [M, N] bfloat16
 */
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
    int t = threadIdx.x;
    int warp_id = t / 32;
    int lane = t % 32;

    int m_start = blockIdx.y * 16;
    int n_start = blockIdx.x * 64;
    int warp_n = n_start + warp_id * 8;

    if (m_start >= M) return;

    int ldm_x = K / 8;
    int ldm_w = K / 8;

    // Accumulator: 4 x int32 per thread (16x8 tile per warp)
    int32_t d[4] = {0, 0, 0, 0};

    for (int k_offset = 0; k_offset < K; k_offset += 64) {
        // --- Load A fragment (4 x uint32) ---
        // Per PTX ISA v9.3: groupID=lane/4, threadID=lane%4
        // a[0]: row=groupID,   K=threadID*8..threadID*8+7   (first half)
        // a[1]: row=groupID+8, K=threadID*8..threadID*8+7   (first half)
        // a[2]: row=groupID,   K=threadID*8+32..threadID*8+7+32 (second half)
        // a[3]: row=groupID+8, K=threadID*8+32..threadID*8+7+32 (second half)
        int a_gid = lane / 4;
        int a_tid = lane % 4;
        uint32_t a[4];
        int x_off = k_offset / 8;
        a[0] = ((const uint32_t*)x_packed)[(m_start + a_gid) * ldm_x + x_off + a_tid];
        a[1] = ((const uint32_t*)x_packed)[(m_start + a_gid + 8) * ldm_x + x_off + a_tid];
        a[2] = ((const uint32_t*)x_packed)[(m_start + a_gid) * ldm_x + x_off + a_tid + 4];
        a[3] = ((const uint32_t*)x_packed)[(m_start + a_gid + 8) * ldm_x + x_off + a_tid + 4];

        // --- Load B fragment (2 x uint32) ---
        // Per PTX ISA v9.3: groupID=lane/4, threadID=lane%4
        // b[0]: col=groupID, K=threadID*8..threadID*8+7     (first half)
        // b[1]: col=groupID, K=threadID*8+32..threadID*8+7+32 (second half)
        int b_gid = lane / 4;
        int b_tid = lane % 4;
        uint32_t b[2];
        int w_off = k_offset / 8;
        b[0] = ((const uint32_t*)w_packed)[(warp_n + b_gid) * ldm_w + w_off + b_tid];
        b[1] = ((const uint32_t*)w_packed)[(warp_n + b_gid) * ldm_w + w_off + b_tid + 4];

        // --- MMA ---
        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1])
        );
    }

    // --- Store accumulator to shared memory ---
    __shared__ float tile[16][64];

    int d_row_a = lane / 4;
    int d_col_pair = (lane % 4) * 2;
    tile[d_row_a][warp_id * 8 + d_col_pair]          = (float)d[0];
    tile[d_row_a][warp_id * 8 + d_col_pair + 1]      = (float)d[1];
    tile[d_row_a + 8][warp_id * 8 + d_col_pair]      = (float)d[2];
    tile[d_row_a + 8][warp_id * 8 + d_col_pair + 1]  = (float)d[3];

    __syncthreads();

    // --- Scale and store to output ---
    // 256 threads cover 16x64 = 1024 elements: 4 elements per thread
    const int total_tile = 16 * 64;
    for (int i = t; i < total_tile; i += blockDim.x) {
        int row = m_start + i / 64;
        int col = n_start + i % 64;
        if (row < M && col < N) {
            float val = tile[i / 64][i % 64] * s_x[row] * w_scale[col];
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
