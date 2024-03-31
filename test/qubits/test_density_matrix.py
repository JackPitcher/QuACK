import sys
sys.path.append(r'c:/Users/jackp/QuACK')

import unittest
import numpy as np
from qubits import StateVector, DensityMatrix
from test_qubit import TestQubit

class TestStateVector(TestQubit):
    
    fp_tol: float = 1e-10
    
    def test_measurement_stats_z_eigenstate(self):
        qubit = DensityMatrix(np.array([[1, 0], [0, 0]]))
        one_state = DensityMatrix(np.array([[0, 0], [0, 1]]))
        zero_state = DensityMatrix(np.array([[1, 0], [0, 0]]))
        measurement_stats = qubit._get_measurement_stats()
        self.assertTrue(DensityMatrix(measurement_stats[0][0]) == zero_state)
        self.assertTrue(DensityMatrix(measurement_stats[0][1]) == one_state)
        self.assertTrue(measurement_stats[1] == [1, 0])
        
    def test_measurement_stats_x_eigenstate(self):
        qubit = DensityMatrix(np.array([[1, 1], [1, 1]]))
        one_state = DensityMatrix(np.array([[0, 0], [0, 1]]))
        zero_state = DensityMatrix(np.array([[1, 0], [0, 0]]))
        measurement_stats = qubit._get_measurement_stats()
        self.assertTrue(DensityMatrix(measurement_stats[0][0]) == zero_state)
        self.assertTrue(DensityMatrix(measurement_stats[0][1]) == one_state)
        self.assertTrue((abs(np.array(measurement_stats[1]) - np.array([0.5, 0.5])) < self.fp_tol).all())
        
    def test_to_state_vector(self):
        one_state = StateVector(np.array([0, 1]))
        zero_state = StateVector(np.array([1, 0]))
        plus_state = StateVector(np.array([1, 1]))
        minus_state = StateVector(np.array([1, -1]))
        yp_state = StateVector(np.array([1, 1j]))
        ym_state = StateVector(np.array([1, -1j]))
        
        one_dm = DensityMatrix(np.array([[0, 0], [0, 1]]))
        zero_dm = DensityMatrix(np.array([[1, 0], [0, 0]]))
        plus_dm = DensityMatrix(np.array([[1, 1], [1, 1]]))
        minus_dm = DensityMatrix(np.array([[1, -1], [-1, 1]]))
        yp_dm = DensityMatrix(np.array([[1, -1j], [1j, 1]]))
        ym_dm = DensityMatrix(np.array([[1, 1j], [-1j, 1]]))
        
        self.assertTrue(StateVector(one_dm.to_state_vector()) == one_state)
        self.assertTrue(StateVector(zero_dm.to_state_vector()) == zero_state)
        self.assertTrue(StateVector(plus_dm.to_state_vector()) == plus_state)
        self.assertTrue(StateVector(minus_dm.to_state_vector()) == minus_state)
        self.assertTrue(StateVector(yp_dm.to_state_vector()) == yp_state)
        self.assertTrue(StateVector(ym_dm.to_state_vector()) == ym_state)
        
    def test_partial_trace(self):
        one_dm = DensityMatrix(np.array([[0, 0], [0, 1]]))
        zero_dm = DensityMatrix(np.array([[1, 0], [0, 0]]))
        plus_dm = DensityMatrix(np.array([[1, 1], [1, 1]]))
        minus_dm = DensityMatrix(np.array([[1, -1], [-1, 1]]))
        yp_dm = DensityMatrix(np.array([[1, -1j], [1j, 1]]))
        ym_dm = DensityMatrix(np.array([[1, 1j], [-1j, 1]]))
        
        op_dm = one_dm.tensor([plus_dm])
        ypz_dm = yp_dm.tensor([zero_dm])
        mym_dm = minus_dm.tensor([ym_dm])
        
        self.assertTrue(DensityMatrix(op_dm.partial_trace(system=1)) == plus_dm)
        self.assertTrue(DensityMatrix(op_dm.partial_trace(system=2)) == one_dm)
        self.assertTrue(DensityMatrix(ypz_dm.partial_trace(system=1)) == zero_dm)
        self.assertTrue(DensityMatrix(ypz_dm.partial_trace(system=2)) == yp_dm)
        self.assertTrue(DensityMatrix(mym_dm.partial_trace(system=1)) == ym_dm)
        self.assertTrue(DensityMatrix(mym_dm.partial_trace(system=2)) == minus_dm)
        
    
if __name__ == '__main__':
    unittest.main()