import numpy as np
import functools as ft
from numba import njit, prange, cuda, complex64


###################
### NUMBA UTILS ###
###################
@njit
def numba_matmul(A: np.array, B: np.array) -> np.array:
    """
    Naive matrix multiplication in Numba.

    Computes C = A @ B

    === Prerequisites ===
    - A.shape[1] == B.shape[0]
    """
    C = np.zeros(shape=(A.shape[0], B.shape[1]), dtype=np.complex64)
    for i in range(A.shape[0]):
        for k in range(A.shape[1]):
            for j in range(B.shape[1]):
                C[i][j] += A[i, k] * B[k, j]
    return C

@njit(parallel=True)
def p_matmul(A: np.array, B: np.array) -> np.array:
    """
    Parallel matrix multiplication in Numba.

    Computes C = A @ B

    === Prerequisites ===
    - A.shape[1] == B.shape[0]
    """
    C = np.zeros(shape=(A.shape[0], B.shape[1]), dtype=np.complex64)
    for i in prange(A.shape[0]):
        for k in prange(A.shape[1]):
            for j in prange(B.shape[1]):
                C[i][j] += A[i, k] * B[k, j]
    return C


def generate_matmul(size: int) -> callable:
    def impl(a, b):
        c = np.zeros((size, size))
        for k in prange(size):
            for i in prange(size):
                for j in prange(size):
                    c[i, j] += a[i, k] * b[k, j]
        return c
    return njit(impl, parallel=True)


def generate_cache_matmul(size: int) -> callable:
    b = size
    def impl(a, b):
        c = np.zeros((size, size))
        for i in prange(size):
            for k in prange(size):
                for j in prange(size):
                    c[i, j] += a[k, i] * b[i, j]
        return c
    return njit(impl, parallel=True)


@njit
def numba_tensor(lst: np.array) -> np.array:
    ret_val = lst[0]
    for ele in lst[1:]:
        ret_val = tensor_prod(ret_val, ele)
    return ret_val

@njit
def tensor_prod(A: np.array, B: np.array) -> np.array:
    m, n = A.shape
    p, q = B.shape
    result = np.zeros((m*p, n*q))
    
    for i in range(m):
        for j in range(n):
            result[i*p:(i+1)*p, j*q:(j+1)*q] = A[i, j] * B
    
    return result

@njit(parallel=True)
def p_tensor_prod(A: np.array, B: np.array) -> np.array:
    m, n = A.shape
    p, q = B.shape
    result = np.zeros((m*p, n*q))
    
    for i in prange(m):
        for j in prange(n):
            result[i*p:(i+1)*p, j*q:(j+1)*q] = A[i, j] * B
    
    return result


##################
### CUDA UTILS ###
##################
@cuda.jit
def cu_conj_T(A: np.array, B: np.array) -> None:
    i, j = cuda.grid(2)
    if i < A.shape[0] and j < A.shape[1]:
        B[i, j] = A[j, i] - 2j * A[j, i].imag


@cuda.jit
def cu_T(A: np.array, B: np.array) -> None:
    i, j = cuda.grid(2)
    if i < A.shape[0] and j < A.shape[1]:
        B[i, j] = A[j, i]


TPB = 4
@cuda.jit
def cu_matmul(A: np.array, B: np.array, C: np.array) -> None:
    """
    Computes C = A @ B. 

    Code from the Numba documentation,
    https://numba.readthedocs.io/en/stable/cuda/examples.html#matrix-multiplication
    """
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=complex64)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=complex64)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = complex64(0.)
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = 0
        sB[tx, ty] = 0
        if x < A.shape[0] and (ty+i*TPB) < A.shape[1]:
          sA[tx, ty] = A[x, ty + i * TPB]
        if y < B.shape[1] and (tx+i*TPB) < B.shape[0]:
          sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()
    if x < C.shape[0] and y < C.shape[1]:
        C[x, y] = tmp


SPB = 8
TPB = 16
@cuda.jit
def cu_left_mul(A: np.array, B: np.array, C: np.array) -> None:
    """
    Computes A @ B[i] for each i in B.
    Adapted from cu_matmul.

    === Prerequisites ===
    - len(B.shape) == 3
    - A.shape == B.shape[1:]
    """
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=complex64)
    sB = cuda.shared.array(shape=(SPB, TPB, TPB), dtype=complex64)

    x, y, z = cuda.grid(3)   
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z
    bpg = cuda.gridDim.y
    
    tmp = complex64(0.)
    for i in range(bpg):
        # Preload data into shared memory
        sA[tz, ty] = 0
        sB[tx, tz, ty] = 0
        if z < A.shape[0] and (ty + i*TPB) < A.shape[1]:
            sA[tz, ty] = A[z, ty + i * TPB]
        if x < B.shape[0] and y < B.shape[2] and (tz + i*TPB) < B.shape[1]:
            sB[tx, tz, ty] = B[x, tz + i * TPB, y]

        cuda.syncthreads()

        # Compute the partial product
        for j in range(TPB):
            tmp += sA[tz, j] * sB[tx, j, ty]

        cuda.syncthreads()
    
    if x < C.shape[0] and z < C.shape[1] and y < C.shape[2]:
        C[x, z, y] = tmp


@cuda.jit
def cu_right_mul(A: np.array, B: np.array, C: np.array) -> None:
    """
    Computes B[i] @ A for each i in B.

    === Prerequisites ===
    - len(B.shape) == 3
    - A.shape == B.shape[1:]
    """
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=complex64)
    sB = cuda.shared.array(shape=(SPB, TPB, TPB), dtype=complex64)

    x, y, z = cuda.grid(3)   
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z
    bpg = cuda.gridDim.y
    
    tmp = complex64(0.)
    for i in range(bpg):
        # Preload data into shared memory
        sB[tx, tz, ty] = 0
        sA[tz, ty] = 0
        if x < B.shape[0] and z < B.shape[1] and (ty + i*TPB) < B.shape[2]:
            sB[tx, tz, ty] = B[x, z, ty + i * TPB]
        if y < A.shape[1] and (tz + i*TPB) < A.shape[0]:
            sA[tz, ty] = A[tz + i * TPB, y]

        cuda.syncthreads()

        # Compute the partial product
        for j in range(TPB):
            tmp += sB[tx, tz, j] * sA[j, ty]

        cuda.syncthreads()
    
    if x < C.shape[0] and z < C.shape[1] and y < C.shape[2]:
        C[x, z, y] = tmp


@cuda.jit
def cu_right_mul_trace(A: np.array, B: np.array, C: np.array) -> None:
    """
    Stores the trace of B[i] @ A^T into C[i], for each i.

    === Prerequisites ===
    - A.shape == B.shape[1:]
    - C.shape == B.shape[0]
    """
    x = cuda.grid(1)
    tmp = complex64(0.)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            tmp += A[i, j] * B[x, i, j]
    C[x] = tmp


@cuda.jit
def cu_trace(A: np.array, B: np.array):
    """
    Stores the trace of A[i] into B[i].

    === Prerequisites ===
    - len(A.shape) == 3
    - A.shape[1] == A.shape[2]
    - A.shape[0] == B.shape[0]
    """
    x = cuda.grid(1)
    if x < A.shape[0]:
        trace = complex64(0.)
        for i in range(A.shape[1]):
            trace += A[x, i, i]
        B[x] = trace

@cuda.jit
def make_copy(A: np.array, B: np.array) -> None:
    """
    Stores A as a copy in B.
    
    === Pre-Requisites ===
    - A.shape = B.shape
    - len(A.shape) == 3
    """
    x, y, z = cuda.grid(3)
    if x < A.shape[0] and y < A.shape[1] and z < A.shape[2]:
        B[x, y, z] = A[x, y, z]

def cuda_tensor(lst: np.array) -> np.array:
    ret_val = lst[0]
    for ele in lst[1:]:
        ret_val = cuda_tensor_prod(ret_val, ele)
    return ret_val


@cuda.jit
def cuda_tensor_prod(A: np.array, B: np.array, C: np.array) -> None:
    i, j = cuda.grid(2)
    if i < A.shape[0] and j < A.shape[1]:
        p, q = B.shape
        C[i*p:(i+1)*p, j*q:(j+1)*q] = A[i, j] * B