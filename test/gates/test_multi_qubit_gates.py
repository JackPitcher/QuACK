import sys
sys.path.append(r'c:/Users/jackp/QuACK')

import unittest
import numpy as np
from gates import SWAP, MultiQubitGate
from qubits import Register, StateVector

class TestMultiQubitGate(unittest.TestCase):
    
    def test_get_state(self):
        zero_state = StateVector([1, 0])
        plus_state = StateVector([1, 1])
        register = Register([zero_state, plus_state], N=4)
        targets = [1, 2]
        gate = MultiQubitGate(register, targets)
        real_state = StateVector(plus_state.tensor(zero_state))
        self.assertTrue(gate.get_state() == real_state)
        
    def test_to_qubit(self):
        one_state = StateVector([0, 1])
        zero_state = StateVector([1, 0])
        register = Register([zero_state, one_state, zero_state, zero_state])
        targets = [0, 1]
        gate = MultiQubitGate(register, targets)
        real_states = [one_state, zero_state]
        test_state = StateVector(one_state.tensor(zero_state))
        self.assertTrue(len(gate.to_qubit(test_state)) == 2)
        for state, real_state in zip(gate.to_qubit(test_state), real_states):   
            self.assertTrue(state == real_state)
        
class TestXGate(TestMultiQubitGate):
    
    def test_matrix_rep(self):
        register = Register(N=2)
        targets = [0, 1]
        gate = SWAP(register, targets)
        real_gate = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        self.assertTrue(np.all(gate.matrix_rep() == real_gate))
        
    def test_evolve_two_qubits(self):
        zero_state = StateVector([1, 0])
        one_state = StateVector([0, 1])
        register = Register([zero_state, one_state])
        targets = [0, 1]
        gate = SWAP(register, targets)
        gate.evolve()
        real_reg = Register([one_state, zero_state])
        for qubit, real_qubit in zip(register, real_reg):
            self.assertTrue(qubit == real_qubit)
            
    def test_evolve_multi_qubits(self):
        zero_state = StateVector([1, 0])
        one_state = StateVector([0, 1])
        register = Register([zero_state, one_state, zero_state, one_state, zero_state])
        targets = [2, 3]
        gate = SWAP(register, targets)
        gate.evolve()
        real_reg = Register([zero_state, one_state, one_state, zero_state, zero_state])
        for qubit, real_qubit in zip(register, real_reg):
            self.assertTrue(qubit == real_qubit)
        

if __name__ == '__main__':
    unittest.main()
    