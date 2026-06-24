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
__launch_bounds__(256, 4)
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

    if (m_start >= M) return;

    int ldm_x = K / 8;
    int ldm_w = K / 8;
    int warp_offset = warp_id * 8;

    int32_t d[4] = {0, 0, 0, 0};

    __shared__ int32_t s_a[16][8 + 1];
    __shared__ int32_t s_b[64][8];

    for (int k_offset = 0; k_offset < K; k_offset += 64) {
        int x_off = k_offset / 8;

        if (t < 128) {
            int row = t / 8, k8 = t % 8;
            s_a[row][k8] = __ldg(&x_packed[(m_start + row) * ldm_x + x_off + k8]);
        }

        int b_col = t / 4;
        int b_k8 = (t % 4) * 2;
        int w_base = (n_start + b_col) * ldm_w + x_off;
        s_b[b_col][b_k8]     = __ldg(&w_packed[w_base + b_k8]);
        s_b[b_col][b_k8 + 1] = __ldg(&w_packed[w_base + b_k8 + 1]);

        __syncthreads();

        int a_gid = lane / 4, a_tid = lane % 4;
        uint32_t a[4];
        a[0] = (uint32_t)s_a[a_gid][a_tid];
        a[1] = (uint32_t)s_a[a_gid + 8][a_tid];
        a[2] = (uint32_t)s_a[a_gid][a_tid + 4];
        a[3] = (uint32_t)s_a[a_gid + 8][a_tid + 4];

        int b_gid = lane / 4, b_tid = lane % 4;
        uint32_t b[2];
        b[0] = (uint32_t)s_b[warp_offset + b_gid][b_tid];
        b[1] = (uint32_t)s_b[warp_offset + b_gid][b_tid + 4];

        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1])
        );

        __syncthreads();
    }

    int groupID = lane / 4;
    int pair = (lane % 4) * 2;
    int col = n_start + warp_id * 8 + pair;
    int row_lo = m_start + groupID;
    int row_hi = m_start + groupID + 8;

    float s_lo = row_lo < M ? __ldg(&s_x[row_lo]) : 0.0f;
    float s_hi = row_hi < M ? __ldg(&s_x[row_hi]) : 0.0f;
    float w0 = col < N ? __ldg(&w_scale[col]) : 0.0f;
    float w1 = col + 1 < N ? __ldg(&w_scale[col + 1]) : 0.0f;

    if (row_lo < M) {
        if (col < N)
            y[row_lo * N + col] = __float2bfloat16((float)d[0] * s_lo * w0);
        if (col + 1 < N)
            y[row_lo * N + col + 1] = __float2bfloat16((float)d[1] * s_lo * w1);
    }
    if (row_hi < M) {
        if (col < N)
            y[row_hi * N + col] = __float2bfloat16((float)d[2] * s_hi * w0);
        if (col + 1 < N)
            y[row_hi * N + col + 1] = __float2bfloat16((float)d[3] * s_hi * w1);
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
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(
            fused_int4_tc_per_token_kernel,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxL1
        );
        attr_set = true;
    }

    dim3 block(256);
    dim3 grid((N + 63) / 64, (M + 15) / 16);
    fused_int4_tc_per_token_kernel<<<grid, block, 0, stream>>>(
        x_packed, w_packed, w_scale, s_x, y, M, N, K
    );
}
