import torch
import triton
import triton.language as tl
@triton.jit
def qk_matmul_kernel(
        Q_ptr, K_ptr, QK_ptr,
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_qkb, stride_qkh, stride_qks, stride_qkd,
        head_dim, seq_len,
        **meta
):
    BLOCK_SIZE_M = meta['BLOCK_SIZE_M']
    BLOCK_SIZE_N = meta['BLOCK_SIZE_N']
    BLOCK_SIZE_K = meta['BLOCK_SIZE_K']

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_bh = tl.program_id(axis=2)

    batch_idx = pid_bh // meta['n_heads']
    head_idx = pid_bh % meta['n_heads']

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    QK = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, head_dim, BLOCK_SIZE_K):
        Q_mask = offs_k[None, :] + i < head_dim
        K_mask = offs_k[:, None] + i < head_dim
        Q_block = tl.load(Q_ptr + batch_idx * stride_qb + head_idx * stride_qh + offs_m[:, None] * stride_qs + (
                    offs_k[None, :] + i) * stride_qd, mask=offs_m[:, None] < seq_len & Q_mask)
        K_block = tl.load(K_ptr + batch_idx * stride_kb + head_idx * stride_kh + offs_n[:, None] * stride_ks + (
                    offs_k[None, :] + i) * stride_kd, mask=offs_n[:, None] < seq_len & K_mask)
        QK += tl.dot(Q_block, tl.trans(K_block))

    QK = QK / tl.sqrt(tl.float32(head_dim))
    QK_ptrs = QK_ptr + batch_idx * stride_qkb + head_idx * stride_qkh + offs_m[:, None] * stride_qks + offs_n[None,
                                                                                                       :] * stride_qkd
    tl.store(QK_ptrs, QK, mask=offs_m[:, None] < seq_len & offs_n[:, None] < seq_len)


### Softmax 内核保持不变
@triton.jit
def softmax_kernel_forward(
        output_ptr,
        input_ptr,
        input_row_stride,
        output_row_stride,
        n_cols,
        causal,
        **meta
):
    row_idx = tl.program_id(0)
    BLOCK_SIZE = meta['BLOCK_SIZE']

    row_start_ptr = input_ptr + row_idx * input_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    mask = col_offsets < n_cols

    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    if causal:
        causal_mask = col_offsets > (row_idx % n_cols)
        row = row + tl.where(causal_mask, -float('inf'), 0.)

    row_minus_max = row - tl.max(row, axis=0)

    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


### 修正后的 Softmax 结果与 V 的矩阵乘法内核

@triton.jit
def softmax_v_matmul_kernel(
        QK_ptr, V_ptr, output_ptr,
        stride_qkb, stride_qkh, stride_qks, stride_qkd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ob, stride_oh, stride_os, stride_od,
        head_dim, seq_len,
        **meta
):
    BLOCK_SIZE_M = meta['BLOCK_SIZE_M']
    BLOCK_SIZE_N = meta['BLOCK_SIZE_N']
    BLOCK_SIZE_K = meta['BLOCK_SIZE_K']

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_bh = tl.program_id(axis=2)

    batch_idx = pid_bh // meta['n_heads']
    head_idx = pid_bh % meta['n_heads']

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    output = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, head_dim, BLOCK_SIZE_K):
        QK_mask = offs_k[None, :] + i < head_dim
        V_mask = offs_k[:, None] + i < head_dim
        QK_block = tl.load(QK_ptr + batch_idx * stride_qkb + head_idx * stride_qkh + offs_m[:, None] * stride_qks + (
                    offs_k[None, :] + i) * stride_qkd, mask=offs_m[:, None] < seq_len & QK_mask)
        V_block = tl.load(
            V_ptr + batch_idx * stride_vb + head_idx * stride_vh + (offs_k[:, None] + i) * stride_vs + offs_n[None,
                                                                                                       :] * stride_vd,
            mask=V_mask & offs_n[:, None] < seq_len)
        output += tl.dot(QK_block, V_block)

    output_ptrs = output_ptr + batch_idx * stride_ob + head_idx * stride_oh + offs_m[:, None] * stride_os + offs_n[None,
                                                                                                            :] * stride_od
    tl.store(output_ptrs, output, mask=offs_m[:, None] < seq_len)


class ChunkScanCombinedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal=False):
        batch_size, n_heads, seq_len, head_dim = Q.shape
        QK = torch.empty((batch_size, n_heads, seq_len, seq_len), device='cuda')
        QK_softmax = torch.empty_like(QK)
        output = torch.empty((batch_size, n_heads, seq_len, head_dim), device='cuda')

        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_K = 32

        grid_qk = (triton.cdiv(seq_len, BLOCK_SIZE_M), triton.cdiv(seq_len, BLOCK_SIZE_N), batch_size * n_heads)
        grid_softmax = (batch_size * n_heads * seq_len, )
        grid_v = (triton.cdiv(seq_len, BLOCK_SIZE_M), triton.cdiv(head_dim, BLOCK_SIZE_N), batch_size * n_heads)

        # QK 矩阵乘法
        qk_matmul_kernel[grid_qk](
            Q, K, QK,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            QK.stride(0), QK.stride(1), QK.stride(2), QK.stride(3),
            head_dim, seq_len,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            n_heads=n_heads
        )

        # Softmax 操作
        softmax_kernel_forward[grid_softmax](
            QK_softmax, QK,
            QK.stride(0), QK.stride(1),
            seq_len,
            causal,
            BLOCK_SIZE=BLOCK_SIZE_M
        )

        # Softmax 结果与 V 的矩阵乘法
        softmax_v_matmul_kernel[grid_v](
            QK_softmax, V, output,
            QK_softmax.stride(0), QK_softmax.stride(1), QK_softmax.stride(2), QK_softmax.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            head_dim, seq_len,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            n_heads=n_heads
            )

        ctx.save_for_backward(Q, K, V, QK_softmax)
        ctx.causal = causal
        return output

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, QK_softmax = ctx.saved_tensors
        causal = ctx.causal

        # Compute gradients here (not implemented in this example)
        grad_Q = torch.zeros_like(Q)
        grad_K = torch.zeros_like(K)
        grad_V = torch.zeros_like(V)

        # For simplicity, return zeros for now
        return grad_Q, grad_K, grad_V, None
