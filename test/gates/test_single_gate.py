import sys
sys.path.append(r'c:/Users/jackp/QuACK')

import unittest
import numpy as np
from gates import X, H, SingleGate
from qubits import Register, StateVector

class TestSingleGate(unittest.TestCase):
    
    def test_get_state(self):
        zero_state = StateVector([1, 0])
        plus_state = StateVector([1, 1])
        register = Register([zero_state, plus_state], N=4)
        targets = [1]
        gate = SingleGate(register, targets)
        self.assertTrue(gate.get_state() == plus_state)
        
    def test_to_qubit(self):
        one_state = StateVector([0, 1])
        zero_state = StateVector([1, 0])
        register = Register([zero_state, one_state, zero_state])
        targets = [1]
        gate = SingleGate(register, targets)
        self.assertTrue(len(gate.to_qubit(one_state)) == 1)
        self.assertTrue(gate.to_qubit(one_state)[0] == zero_state)
        
class TestXGate(TestSingleGate):
    
    def test_matrix_rep(self):
        register = Register(N=1)
        targets = [0]
        gate = X(register, targets)
        real_gate = np.array([[0, 1], [1, 0]])
        self.assertTrue(np.all(gate.matrix_rep() == real_gate))
        
    def test_evolve_one_qubit(self):
        register = Register(N = 1)
        targets = [0]
        gate = X(register, targets)
        gate.evolve()
        one_state = StateVector([1, 0])
        for qubit in register:
            self.assertTrue(qubit == one_state)
            
    def test_evolve_two_qubit(self):
        register = Register(N = 3)
        targets = [1]
        gate = X(register, targets)
        gate.evolve()
        zero_state = StateVector([1, 0])
        one_state = StateVector([0, 1])
        self.assertTrue(register[0] == zero_state)
        self.assertTrue(register[1] == one_state)
        self.assertTrue(register[2] == zero_state)
        
class TestHGate(TestSingleGate):
    
    def test_matrix_rep(self):
        register = Register(N=1)
        targets = [0]
        gate = H(register, targets)
        real_gate = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
        self.assertTrue(np.all(gate.matrix_rep() == real_gate))
        
    def test_evolve_one_qubit(self):
        register = Register(N = 1)
        targets = [0]
        gate = H(register, targets)
        gate.evolve()
        plus_state = StateVector([1, 1])
        for qubit in register:
            self.assertTrue(qubit == plus_state)
            
    def test_evolve_two_qubit(self):
        register = Register(N = 3)
        targets = [1]
        gate = H(register, targets)
        gate.evolve()
        zero_state = StateVector([1, 0])
        plus_state = StateVector([1, 1])
        self.assertTrue(register[0] == zero_state)
        self.assertTrue(register[1] == plus_state)
        self.assertTrue(register[2] == zero_state)
        
        


if __name__ == '__main__':
    unittest.main()
    