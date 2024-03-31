import sys
sys.path.append(r'c:/Users/jackp/QuACK')

import unittest
import numpy as np
from qubits import StateVector, Register, DensityMatrix

class TestRegister(unittest.TestCase):
    
    def test_init_n_parameter(self):
        reg = Register(N=2)
        ground_state = DensityMatrix([[1, 0], [0, 0]])
        self.assertTrue(len(reg) == 2)
        self.assertTrue(reg.state == DensityMatrix(np.kron(ground_state.get_state(), ground_state.get_state())))
        for qubit in reg:
            self.assertTrue(qubit == ground_state)
    
    def test_init_state_vector_parameter(self):
        qubits = [StateVector([1, 0]), StateVector([0, 1]), StateVector([1, 1])]
        reg = Register(qubits=qubits.copy())
        self.assertTrue(len(reg) == 3)
        zero_state = np.array([[1, 0], [0, 0]])
        one_state = np.array([[0, 0], [0, 1]])
        plus_state = 0.5 * np.array([[1, 1], [1, 1]])
        self.assertTrue(reg.state == DensityMatrix(np.kron(zero_state, np.kron(one_state, plus_state))))
        for real_qubit, test_qubit in zip(qubits, reg):
            self.assertTrue(DensityMatrix(real_qubit.to_density_matrix()) == test_qubit)
            
    def test_init_two_parameters(self):
        qubits = [StateVector([1, 0]), StateVector([1, 1])]
        reg = Register(qubits=qubits.copy(), N=4)
        qubits.extend([StateVector([0, 1]), StateVector([0, 1])])
        zero_state = np.array([[1, 0], [0, 0]])
        plus_state = 0.5 * np.array([[1, 1], [1, 1]])
        self.assertTrue(len(reg) == 4)
        self.assertTrue(reg.state == DensityMatrix(np.kron(zero_state, np.kron(plus_state, np.kron(zero_state, zero_state)))))
        for real_qubit, test_qubit in zip(qubits, reg):
            self.assertTrue(DensityMatrix(real_qubit.to_density_matrix()) == test_qubit)
    
    def test_add_qubit(self):
        qubits = [StateVector([1, 0]), StateVector([0, 1])]
        reg = Register(qubits=qubits.copy())
        self.assertTrue(len(reg) == 2)
        reg.add_qubit(StateVector([1, 1]))
        qubits.append(StateVector([1, 1]))
        self.assertTrue(len(reg) == 3)
        for real_qubit, test_qubit in zip(qubits, reg):
            self.assertTrue(DensityMatrix(real_qubit.to_density_matrix()) == test_qubit)
    
    def test_remove_qubit(self):
        qubits = [StateVector([1, 0]), StateVector([0, 1]), StateVector([1, 1])]
        reg = Register(qubits=qubits.copy())
        self.assertTrue(len(reg) == 3)
        self.assertTrue(reg.remove_qubit(StateVector([1, 1])) is None)
        qubits.pop(-1)
        zero_state = np.array([[1, 0], [0, 0]])
        one_state = np.array([[0, 0], [0, 1]])
        self.assertTrue(reg.state == DensityMatrix(np.kron(zero_state, one_state)))
        self.assertTrue(len(reg) == 2)
        for real_qubit, test_qubit in zip(qubits, reg):
            self.assertTrue(DensityMatrix(real_qubit.to_density_matrix()) == test_qubit)
            
    def test_remove_index(self):
        qubits = [StateVector([1, 0]), StateVector([0, 1]), StateVector([1, 1])]
        reg = Register(qubits=qubits.copy())
        self.assertTrue(len(reg) == 3)
        test_remove = reg.remove_qubit(-1)
        real_remove = qubits.pop(-1)
        self.assertTrue(DensityMatrix(real_remove.to_density_matrix()) == test_remove)
        self.assertTrue(len(reg) == 2)
        zero_state = np.array([[1, 0], [0, 0]])
        one_state = np.array([[0, 0], [0, 1]])
        self.assertTrue(reg.state == DensityMatrix(np.kron(zero_state, one_state)))
        for real_qubit, test_qubit in zip(qubits, reg):
            self.assertTrue(DensityMatrix(real_qubit.to_density_matrix()) == test_qubit)
    
    def test_measure_one_state(self):
        qubits = [StateVector([1, 0]), StateVector([0, 1]), StateVector([1, 1])]
        reg = Register(qubits=qubits)
        one_state = StateVector([0, 1])
        reg.measure(index=1, return_stats=False)
        self.assertTrue(DensityMatrix(one_state.to_density_matrix()) == reg[1])
        
    def test_measure_zero_state(self):
        qubits = [StateVector([1, 0]), StateVector([0, 1]), StateVector([1, 1])]
        reg = Register(qubits=qubits)
        zero_state = StateVector([1, 0])
        reg.measure(index=0, return_stats=False)
        self.assertTrue(DensityMatrix(zero_state.to_density_matrix()) == reg[0])
        
    def test_measure_plus_state(self):
        qubits = [StateVector([1, 0]), StateVector([0, 1]), StateVector([1, 1])]
        reg = Register(qubits=qubits)
        zero_state = StateVector([1, 0])
        one_state = StateVector([0, 1])
        reg.measure(index=2, return_stats=False)
        self.assertTrue(DensityMatrix(zero_state.to_density_matrix()) == reg[2] 
                        or DensityMatrix(one_state.to_density_matrix()) == reg[2])
    
    # def test_reorder(self):
    #     qubits = [StateVector([1, 0]), StateVector([0, 1]), StateVector([1, 1])]
    #     reg = Register(qubits=qubits.copy())
    #     self.assertTrue(len(reg) == 3)
    #     for real_qubit, test_qubit in zip(qubits, reg):
    #         self.assertTrue(DensityMatrix(real_qubit.to_density_matrix()) == test_qubit)
            
    #     reordered_qubits = [StateVector([1, 1]), StateVector([1, 0]), StateVector([0, 1])]
    #     reg.reorder([2, 0, 1])
    #     for real_qubit, test_qubit in zip(reordered_qubits, reg):
    #         self.assertTrue(real_qubit.to_density_matrix() == test_qubit)
    
    # def test_array_rep(self):
    #     qubits = [StateVector([1j, 0]), StateVector([0, -1j]), StateVector([1, 0])]
    #     reg = Register(qubits=qubits.copy())
    #     proper_return = np.array([1j, 0, 0, -1j, 1, 0])[..., None]
    #     self.assertTrue((reg.array_representation() == proper_return).all())
        
    # def test_array_rep_bra(self):
    #     qubits = [StateVector([1j, 0]), StateVector([0, -1j]), StateVector([1, 0])]
    #     reg = Register(qubits=qubits.copy())
    #     proper_return = np.array([-1j, 0, 0, 1j, 1, 0])
    #     self.assertTrue((reg.array_representation(return_ket=False) == proper_return).all())
    
if __name__ == '__main__':
    unittest.main()