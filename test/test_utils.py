import sys
sys.path.append(r'c:/Users/jackp/QuACK')
sys.path.append(r'c:/Users/npmgs/github/QuACK')

import unittest
import numpy as np
from numba import cuda
import math
import utils

class TestUtils(unittest.TestCase):
    tol = 1e-6
    def test_cu_conj_T(self):
        A = np.array([[1, 2 + 1j], [3, 4 - 5j]], dtype=np.complex64)
        B = np.zeros_like(A)
        
        threadsperblock = (16, 16)
        blockspergrid = (1, 1)
        utils.cu_conj_T[blockspergrid, threadsperblock](A, B)
        expected = np.array([[1, 3], [2-1j, 4 + 5j]], dtype=np.complex64)
        assert np.linalg.norm(B - expected) < self.tol

    def test_cu_left_mul(self):
        A = np.array([[2, 0], [0,3]], dtype=np.complex64)
        B = np.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ], dtype=np.complex64)
        C = np.zeros_like(B)
        threadsperblock = (4,4,4)
        bpg_x = math.ceil(B.shape[0] / threadsperblock[0])
        bpg_y = math.ceil(B.shape[1] / threadsperblock[1])
        bpg_z = math.ceil(B.shape[2] / threadsperblock[2])
        blockspergrid = (bpg_x, bpg_y, bpg_z)
        utils.cu_left_mul[blockspergrid, threadsperblock](A, B, C)

        expected = np.array([
            [[2, 4], [9, 12]],
            [[10, 12], [21, 24]]
        ], dtype=np.complex64)

        assert np.linalg.norm(C - expected) < self.tol

    def test_cu_right_mul(self):
        A = np.array([[2, 0], [0,3]], dtype=np.complex64)
        B = np.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ], dtype=np.complex64)
        C = np.zeros_like(B)
        threadsperblock = (4,4,4)
        bpg_x = math.ceil(B.shape[0] / threadsperblock[0])
        bpg_y = math.ceil(B.shape[1] / threadsperblock[1])
        bpg_z = math.ceil(B.shape[2] / threadsperblock[2])
        blockspergrid = (bpg_x, bpg_y, bpg_z)
        utils.cu_right_mul[blockspergrid, threadsperblock](A, B, C)

        expected = np.array([
            [[2, 6], [6, 12]],
            [[10, 18], [14, 24]]
        ], dtype=np.complex64)
        assert np.linalg.norm(C - expected) < self.tol

    def test_mul_large(self):
        init_A = np.arange(64, dtype=np.complex64).reshape(8, 8)
        A = init_A.copy()
        init_B = np.arange(256, dtype=np.complex64).reshape(-1, 8, 8)
        B = init_B.copy()
        C = np.zeros_like(B)

        threadsperblock = (8,8,8)
        bpg_x = math.ceil(B.shape[0] / threadsperblock[0])
        bpg_y = math.ceil(B.shape[1] / threadsperblock[1])
        bpg_z = math.ceil(B.shape[2] / threadsperblock[2])
        blockspergrid = (bpg_x, bpg_y, bpg_z)

        utils.cu_left_mul[blockspergrid, threadsperblock](A, B, C)
        expected = np.zeros_like(B)
        for i in range(B.shape[0]):
            expected[i] = A @ B[i]

        assert np.linalg.norm(A - init_A) < self.tol
        assert np.linalg.norm(B - init_B) < self.tol
        assert np.linalg.norm(C - expected) < self.tol

        utils.cu_right_mul[blockspergrid, threadsperblock](A, B, C)
        expected = np.zeros_like(B)
        for i in range(B.shape[0]):
            expected[i] = B[i] @ A
        assert np.linalg.norm(A - init_A) < self.tol
        assert np.linalg.norm(B - init_B) < self.tol
        assert np.linalg.norm(C - expected) < self.tol

    def test_cu_right_mul_trace(self):
        A = np.arange(16, dtype=np.complex64).reshape(4, 4)
        B = np.arange(320, dtype=np.complex64).reshape(20, 4, 4)
        C = np.zeros(20, dtype=np.complex64)

        threadsperblock = 32
        blockspergrid = math.ceil(A.shape[0] / threadsperblock)

        utils.cu_right_mul_trace[blockspergrid, threadsperblock](A, B, C)
        expected = np.zeros(20, dtype=np.complex64)
        for i in range(B.shape[0]):
            expected[i] = np.trace(B[i] @ A.T)
        
        assert np.linalg.norm(C - expected) < self.tol

    def test_cu_trace(self):
        A = np.arange(1280, dtype=np.complex64).reshape(20, 8, 8)
        B = np.zeros(20, dtype=np.complex64)

        threadsperblock = 32
        blockspergrid = math.ceil(A.shape[0] / threadsperblock)

        utils.cu_trace[blockspergrid, threadsperblock](A, B)
        expected = np.zeros(20, dtype=np.complex64)
        for i in range(A.shape[0]):
            expected[i] = np.trace(A[i])
        
        assert np.linalg.norm(B - expected) < self.tol


if __name__ == '__main__':
    unittest.main()