import sys
sys.path.append(r'c:/Users/jackp/QuACK')

import unittest
import numpy as np
from qubits import StateVector, DensityMatrix
from test_qubit import TestQubit

class TestStateVector(TestQubit):
    
    fp_tol: float = 1e-10
        
    def test_bra_real(self):
        qubit = StateVector(np.array([1, 1]))
        bra = 1 / np.sqrt(2) * np.array([1, 1])
        test_bra = qubit.bra()
        self.assertTrue(test_bra.shape == (2,))
        self.assertTrue((abs(test_bra - bra) < self.fp_tol).all())
    
    def test_bra_imaginary(self):
        qubit = StateVector(np.array([1, 1j]))
        bra = 1 / np.sqrt(2) * np.array([1, -1j])
        test_bra = qubit.bra()
        self.assertTrue(test_bra.shape == (2,))
        self.assertTrue((abs(test_bra - bra) < self.fp_tol).all())
        
    def test_bra_complex(self):
        qubit = StateVector(np.array([1 + 1j, 1 + 1j]))
        bra = 1/2 * np.array([1 - 1j, 1 - 1j])
        test_bra = qubit.bra()
        self.assertTrue(test_bra.shape == (2,))
        self.assertTrue((abs(test_bra - bra) < self.fp_tol).all())
    
    def test_ket_real(self):
        qubit = StateVector(np.array([1, 1]))
        ket = 1 / np.sqrt(2) * np.array([[1], [1]])
        test_ket = qubit.ket()
        self.assertTrue(test_ket.shape == (2,1))
        self.assertTrue((abs(test_ket - ket) < self.fp_tol).all())
        
    def test_ket_imaginary(self):
        qubit = StateVector(np.array([1, 1j]))
        ket = 1 / np.sqrt(2) * np.array([[1], [1j]])
        test_ket = qubit.ket()
        self.assertTrue(test_ket.shape == (2,1))
        self.assertTrue((abs(test_ket - ket) < self.fp_tol).all())
        
    def test_ket_complex(self):
        qubit = StateVector(np.array([1 + 1j, 1 + 1j]))
        ket = 1/2 * np.array([[1 + 1j], [1 + 1j]])
        test_ket = qubit.ket()
        self.assertTrue(test_ket.shape == (2,1))
        self.assertTrue((abs(test_ket - ket) < self.fp_tol).all())
    
    def test_measurement_stats_z_eigenstate(self):
        qubit = StateVector(np.array([1, 0]))
        one_state = StateVector(np.array([0, 1]))
        zero_state = StateVector(np.array([1, 0]))
        measurement_stats = qubit._get_measurement_stats()
        self.assertTrue((abs(measurement_stats[0][1].ket() - one_state.ket()) < self.fp_tol).all())
        self.assertTrue((abs(measurement_stats[0][0].ket() - zero_state.ket()) < self.fp_tol).all())
        self.assertTrue(measurement_stats[1] == [1, 0])
        
    def test_measurement_stats_x_eigenstate(self):
        qubit = StateVector(np.array([1, 1]))
        one_state = StateVector(np.array([0, 1]))
        zero_state = StateVector(np.array([1, 0]))
        measurement_stats = qubit._get_measurement_stats()
        self.assertTrue((abs(measurement_stats[0][1].ket() - one_state.ket()) < self.fp_tol).all())
        self.assertTrue((abs(measurement_stats[0][0].ket() - zero_state.ket()) < self.fp_tol).all())
        self.assertTrue((abs(np.array(measurement_stats[1]) - np.array([0.5, 0.5])) < self.fp_tol).all())
        
    def test_to_density_matrix(self):
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
        
        self.assertTrue(one_dm == DensityMatrix(one_state.to_density_matrix()))
        self.assertTrue(zero_dm == DensityMatrix(zero_state.to_density_matrix()))
        self.assertTrue(plus_dm == DensityMatrix(plus_state.to_density_matrix()))
        self.assertTrue(minus_dm == DensityMatrix(minus_state.to_density_matrix()))
        self.assertTrue(yp_dm == DensityMatrix(yp_state.to_density_matrix()))
        self.assertTrue(ym_dm == DensityMatrix(ym_state.to_density_matrix()))
    
if __name__ == '__main__':
    unittest.main()