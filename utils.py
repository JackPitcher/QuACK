import numpy as np
import functools as ft
from numba import njit, prange, cuda

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

@cuda.jit
def cuda_matmul(A: np.array, B: np.array, C: np.array) -> np.array:
    """
    Naive matrix multiplication in Numba CUDA.
    
    Computes C = A @ B

    === Prerequisites ===
    - A.shape[1] == B.shape[0]
    - C.shape[0] == A.shape[0]
    - C.shape[1] == B.shape[1]
    """
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0j
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp

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
