import triton
import triton.language as tl
import torch
@triton.jit
def flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, output_ptr,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_vb, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_od,
    n_heads, seq_len, head_dim,
    **meta
):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)

    offs_m = pid * meta['BLOCK_SIZE_M'] + tl.arange(0, meta['BLOCK_SIZE_M'])
    offs_n = bid * meta['BLOCK_SIZE_N'] + tl.arange(0, meta['BLOCK_SIZE_N'])
    offs_k = tl.arange(0, head_dim)

    Q = tl.load(Q_ptr + (offs_m[:, None] * stride_qh + offs_k[None, :] * stride_qd), mask=offs_m[:, None] < seq_len)
    K = tl.load(K_ptr + (offs_n[:, None] * stride_kh + offs_k[None, :] * stride_kd), mask=offs_n[:, None] < seq_len)
    V = tl.load(V_ptr + (offs_n[:, None] * stride_vh + offs_k[None, :] * stride_vd), mask=offs_n[:, None] < seq_len)

    QK = tl.dot(Q, K)
    QK = QK / tl.sqrt(tl.float32(head_dim))
    QK = tl.softmax(QK, axis=1)

    output = tl.dot(QK, V)
    output_ptrs = output_ptr + (offs_m[:, None] * stride_ob + offs_n[None, :] * stride_oh)
    tl.store(output_ptrs, output, mask=offs_m[:, None] < seq_len)

def flash_attention(Q, K, V, n_heads, seq_len, head_dim):
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128

    grid = lambda meta: (triton.cdiv(seq_len, meta['BLOCK_SIZE_M']), triton.cdiv(seq_len, meta['BLOCK_SIZE_N']))

    output = torch.empty_like(Q)
    flash_attention_kernel[grid](
        Q, K, V, output,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        n_heads, seq_len, head_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    return output