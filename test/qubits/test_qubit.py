import sys
sys.path.append(r'c:/Users/jackp/QuACK')

import unittest
import numpy as np
from qubits import StateVector, DensityMatrix

class TestQubit(unittest.TestCase):
    
    fp_tol: float = 1e-10
        
    def test_init_normalized(self):
        qubit_state = 1 / np.sqrt(2) * np.array([1, 1])
        qubit = StateVector(qubit_state)
        self.assertTrue((abs(qubit._state - qubit_state) < self.fp_tol).all())
        
        dm_state = 1 / 2 * np.array([[1, 1], [1, 1]])
        dm = DensityMatrix(dm_state)
        self.assertTrue((abs(dm._state - dm_state) < self.fp_tol).all())
        
    def test_init_unnormalized(self):
        qubit_state = np.array([1, 1])
        qubit = StateVector(qubit_state)
        self.assertTrue((abs(qubit._state - 1 / np.sqrt(2) * qubit_state) < self.fp_tol).all())
        
        dm_state = np.array([[1, 1], [1, 1]])
        dm = DensityMatrix(dm_state)
        self.assertTrue((abs(dm._state - 1 / 2 * dm_state) < self.fp_tol).all())
        
    def test_tensor(self):
        one_state = StateVector(np.array([0, 1]))
        zero_state = StateVector(np.array([1, 0]))
        plus_state = StateVector(np.array([1, 1]))
        minus_state = StateVector(np.array([1, -1]))
        
        one_dm = DensityMatrix(np.array([[0, 0], [0, 1]]))
        zero_dm = DensityMatrix(np.array([[1, 0], [0, 0]]))
        plus_dm = DensityMatrix(np.array([[1, 1], [1, 1]]))
        minus_dm = DensityMatrix(np.array([[1, -1], [-1, 1]]))
        
        oo_state = StateVector(np.array([0, 0, 0, 1]))
        zo_state = StateVector(np.array([0, 1, 0, 0]))
        oz_state = StateVector(np.array([0, 0, 1, 0]))
        zz_state = StateVector(np.array([1, 0, 0, 0]))
        pm_state = StateVector( 0.5 * np.array([1, -1, 1, -1]))
        ooo_state = StateVector(np.array([0, 0, 0, 0, 0, 0, 0, 1]))
        pzo_state = StateVector(np.array([0, 1 / np.sqrt(2), 0, 0, 0, 1 / np.sqrt(2), 0, 0]))
        
        oo_dm = DensityMatrix(np.array([[0, 0, 0, 0],
                                        [0, 0, 0, 0],
                                        [0, 0, 0, 0],
                                        [0, 0, 0, 1]]))
        zo_dm = DensityMatrix(np.array([[0, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 0, 0],
                                        [0, 0, 0, 0]]))
        oz_dm = DensityMatrix(np.array([[0, 0, 0, 0],
                                        [0, 0, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 0]]))
        zz_dm = DensityMatrix(np.array([[1, 0, 0, 0],
                                        [0, 0, 0, 0],
                                        [0, 0, 0, 0],
                                        [0, 0, 0, 0]]))
        pm_dm = DensityMatrix(0.5 * np.array([[ 1, -1,  1, -1],
                                        [-1,  1, -1,  1],
                                        [ 1, -1,  1, -1],
                                        [-1,  1, -1,  1]]))
        ooo_dm = DensityMatrix(np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 1]]))
        pzo_dm = DensityMatrix(0.5 * np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 1, 0, 0, 0, 1, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 1, 0, 0, 0, 1, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0]]))
        
        self.assertTrue(oo_state == StateVector(one_state.tensor(one_state)))
        self.assertTrue(zo_state == StateVector(zero_state.tensor(one_state)))
        self.assertTrue(oz_state == StateVector(one_state.tensor(zero_state)))
        self.assertTrue(zz_state == StateVector(zero_state.tensor(zero_state)))
        self.assertTrue(pm_state == StateVector(plus_state.tensor(minus_state)))
        self.assertTrue(ooo_state == StateVector(one_state.tensor([one_state, one_state])))
        self.assertTrue(pzo_state == StateVector(plus_state.tensor([zero_state, one_state])))
        
        self.assertTrue(oo_dm == DensityMatrix(one_dm.tensor(one_dm)))
        self.assertTrue(zo_dm == DensityMatrix(zero_dm.tensor(one_dm)))
        self.assertTrue(oz_dm == DensityMatrix(one_dm.tensor(zero_dm)))
        self.assertTrue(zz_dm == DensityMatrix(zero_dm.tensor(zero_dm)))
        self.assertTrue(pm_dm == DensityMatrix(plus_dm.tensor(minus_dm)))
        self.assertTrue(ooo_dm == DensityMatrix(one_dm.tensor([one_dm, one_dm])))
        self.assertTrue(pzo_dm == DensityMatrix(plus_dm.tensor([zero_dm, one_dm])))
        
    def test_measure_zero(self):
        qubit = StateVector(np.array([1, 0]))
        zero_state = StateVector(np.array([1, 0]))
        qubit.measure()
        self.assertTrue(qubit == zero_state)
        
        dm = DensityMatrix(np.array([[1, 0], [0, 0]]))
        zero_dm = DensityMatrix(np.array([[1, 0], [0, 0]]))
        dm.measure()
        self.assertTrue(dm == zero_dm)
        
    def test_measure_one(self):
        qubit = StateVector(np.array([0, 1]))
        one_state = StateVector(np.array([0, 1]))
        qubit.measure()
        self.assertTrue(qubit == one_state)
        
        dm = DensityMatrix(np.array([[0, 0], [0, 1]]))
        one_dm = DensityMatrix(np.array([[0, 0], [0, 1]]))
        dm.measure()
        self.assertTrue(dm == one_dm)
        
    def test_measure_plus(self):
        qubit = StateVector(np.array([1, 1]))
        one_state = StateVector(np.array([0, 1]))
        zero_state = StateVector(np.array([1, 0]))
        qubit.measure()
        self.assertTrue((qubit == one_state or qubit == zero_state))
        
        dm = DensityMatrix(np.array([[1, 1], [1, 1]]))
        one_dm = DensityMatrix(np.array([[0, 0], [0, 1]]))
        zero_dm = DensityMatrix(np.array([[1, 0], [0, 0]]))
        dm.measure()
        self.assertTrue(dm == one_dm or dm == zero_dm)
    
if __name__ == '__main__':
    unittest.main()