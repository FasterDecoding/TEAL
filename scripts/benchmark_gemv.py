import os

import torch
import triton

import numpy as np

import sys

import sys
sys.path.append("../")
from kernels.sparse_gemv import splitk_sparse_gemv, splitk_sparse_gemv_kernel

from typing import Optional

import torch
import triton
import triton.language as tl
def init_to_zero(*names):
    def init_func(nargs):
        for name in names:
            nargs[name].zero_()
    return init_func

# 20 bins
# cache the sparsity level so autotune sees it
def discretize_sparsity(x):
    # return 0 # comment out if don't want super long tuning
    # return 0
    return int(abs(x) // 0.05)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_warps=2, pre_hook=init_to_zero("Y")),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 16}, num_warps=4, pre_hook=init_to_zero("Y")),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, pre_hook=init_to_zero("Y")),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, pre_hook=init_to_zero("Y")),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, pre_hook=init_to_zero("Y")),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")),
    ],
    key=["CACHE_KEY_M", "CACHE_KEY_N", "BATCHSIZE", "SPARSITY_BIN"],
)
@triton.jit
def gather_transposed_gemv_flag_atomicadd_kernel(
    Y,  # Pointers to matrices
    A,
    X,
    IDX,
    # Matrix dimensions
    M,
    N,
    CACHE_KEY_M,
    CACHE_KEY_N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am,
    # Meta-parameters
    BATCHSIZE: tl.constexpr,
    SPARSITY_BIN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):

    """
    Kernel for computing Y = A[IDX, :]^T @ X + BIAS, where A is a dense matrix
    with Z rows and N columns. We also batch across the batch dimension of the input X.
    We will not check that the indices are valid, for performance reason.
    - Input X has shape (BATCHSIZE, M)
    - Weight has shape (Z, N)
    - IDX has shape (M), where M is the number of non-zero rows in A
    - Bias has shape (N)
    - Output has shape (BATCHSIZE, N)
    """
    start_m = tl.program_id(0)
    start_n = tl.program_id(1)
    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices for rows (resp. col) of A
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    IDX = IDX + rm
    idx = tl.load(IDX, mask=rm < M, other=0) > 0
    A = A + (rm[:, None] * stride_am + rn[None, :])
    X = X + rm
    Y = Y + rn
    
    if BATCHSIZE == 1:
        a = tl.load(A, mask=idx[:, None], other=0.0)
        x0 = tl.load(X)#, mask=idx, other=0.0) # if flag_gemv is correct, this will be unnecessary.
        acc0 = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)

    # rematerialize rm and rn to save registers
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.atomic_add(Y, acc0, mask=rn < N)
    
def gather_transposed_gemv_flag_3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    idx: torch.Tensor,
    sparsity_bin: int
) -> torch.Tensor:
    """
    Compute y = weight[idx, :]^T @ x.
    :param x: input tensor
    :param weight: weight matrix
    :param idx: indices
    :return: result tensor
    """
    Z, N = weight.shape
    beam_width, seq_len, _ = x.shape
    assert x.shape[2] == Z
    x = x.contiguous()
    if weight.stride(1) > 1:
        weight = weight.contiguous()

    output = torch.empty(
        beam_width,
        seq_len,
        N,
        device=x.device,
        dtype=torch.float32,
    )

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(Z, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )  # noqa

    kernel = gather_transposed_gemv_flag_atomicadd_kernel
    kernel[grid](
        output,  # data ptrs
        weight,
        x,
        idx,
        Z,  # shapes
        N,
        Z // 128,  # key for triton cache (limit number of compilations)
        N // 32,
        weight.stride(0),  # strides
        beam_width,  # can't use kwargs because auto-tuner requires args
        sparsity_bin,
    )
    return output# .to(dtype=weight.dtype)

# maybe not column major?
def deja_vu_gemv(x, weight, sparsity_level):
    idx = x.abs() > sparsity_level/2
    return gather_transposed_gemv_flag_3d(x, weight, idx, discretize_sparsity(sparsity_level))

def our_sparse_gemv(x, weight, sparsity_level):
    # assuming uniform random
    threshold = sparsity_level / 2

    return splitk_sparse_gemv(x, weight, threshold, discretize_sparsity(sparsity_level))

def dense_gemv(x, weight):
    return x @ weight

# 20 bins
# cache the sparsity level so autotune sees it
def discretize_sparsity(x):
    # return 0 # comment out if don't want super long tuning
    # return 0
    return int(abs(x) // 0.05)


zeal_results = []
deja_vu_results = []
dense_results = []
theoretical_results = []

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['sparsity_level'],  # Argument names to use as an x-axis for the plot.
        x_vals=[i*0.01 for i in range(0, 101)],  # Different possible values for `x_name`.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['zeal', 'deja vu', 'dense', 'theoretical optimal'],  # Possible values for `line_arg`.
        line_names=['ZEAL', 'Deja Vu', 'Dense', 'Theoretical Optimal'],  # Label name for the lines.
        styles=[('blue', '-'),('purple', '-'),  ('green', '-'), ('red', '-')],  # Line styles.
        ylabel='ms',  # Label name for the y-axis.
        args={"in_size": 4096, "out_size": 14336},  # Values for function arguments not in `x_names` and `y_name`.
        plot_name='Kernel Plot (A6000) (4096x14336)',  # Name for the plot. Used also as a file name for saving the plot.
    ))
def benchmark(sparsity_level, provider, in_size, out_size):
    x = torch.rand((1, 1, in_size), device='cuda', dtype=torch.float16) - 0.5
    W = torch.rand((out_size, in_size), device='cuda', dtype=torch.float16) - 0.5

    W = W.T.contiguous().T
    W_T = W.T.contiguous()
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'dense':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: dense_gemv(x, W_T), quantiles=quantiles)
        dense_results.append((sparsity_level, ms, min_ms, max_ms))

    if provider == 'zeal':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: our_sparse_gemv(x, W, sparsity_level), quantiles=quantiles, rep=1000)
        print(splitk_sparse_gemv_kernel.best_config, ms)
        zeal_results.append((sparsity_level, ms, min_ms, max_ms))
    if provider == 'deja vu':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: deja_vu_gemv(x, W_T, sparsity_level), quantiles=quantiles, rep=1000)
        # print(gather_transposed_gemv_flag_atomicadd_kernel.best_config, ms)
        deja_vu_results.append((sparsity_level, ms, min_ms, max_ms))
    if provider == 'theoretical optimal':   
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: dense_gemv(x, W_T), quantiles=quantiles)
        ms *= (1 - sparsity_level)
        min_ms *= (1 - sparsity_level)
        max_ms *= (1 - sparsity_level)
        theoretical_results.append((sparsity_level, ms, min_ms, max_ms))

    return ms, max_ms, min_ms

    # gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6
    # return gbps(ms), gbps(max_ms), gbps(min_ms)

print("running...")
save_path = "./benchmark_results"
os.makedirs(save_path, exist_ok=True)

# Run the benchmark
benchmark.run(print_data=True, show_plots=True)

# Save results to CSV
import pandas as pd

df = pd.DataFrame({
    'sparsity_level': [r[0] for r in zeal_results],
    'ZEAL': [r[1] for r in zeal_results],
    'ZEAL_min': [r[2] for r in zeal_results],
    'ZEAL_max': [r[3] for r in zeal_results],
    'Deja Vu': [r[1] for r in deja_vu_results],
    'Deja Vu_min': [r[2] for r in deja_vu_results],
    'Deja Vu_max': [r[3] for r in deja_vu_results],
    'Dense': [r[1] for r in dense_results],
    'Dense_min': [r[2] for r in dense_results],
    'Dense_max': [r[3] for r in dense_results],
    'Theoretical Optimal': [r[1] for r in theoretical_results],
    'Theoretical Optimal_min': [r[2] for r in theoretical_results],
    'Theoretical Optimal_max': [r[3] for r in theoretical_results],
})

df.to_csv(f"{save_path}/Kernel Plot (A100) (4096x14336).csv", index=False)

print(f"Results saved to {save_path}/Kernel Plot (A100) (4096x14336).csv")