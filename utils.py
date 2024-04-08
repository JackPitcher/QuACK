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


TPB = 4
@cuda.jit
def cu_left_mul(A: np.array, B: np.array, C: np.array) -> None:
    """
    Computes A @ B[i] for each i in B.

    === Prerequisites ===
    - len(B.shape) == 3
    - A.shape == B.shape[1:]
    """
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=complex64)
    sB = cuda.shared.array(shape=(TPB, TPB, TPB), dtype=complex64)

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
    sB = cuda.shared.array(shape=(TPB, TPB, TPB), dtype=complex64)

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