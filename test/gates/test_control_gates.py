import sys
sys.path.append(r'c:/Users/jackp/QuACK')

import unittest
import numpy as np
from gates import ControlledGate, CNOT, CSWAP
from qubits import Register, StateVector, DensityMatrix

class TestMultiQubitGate(unittest.TestCase):
    
    def test_get_state_one_target_one_control(self):
        zero_state = StateVector([1, 0])
        plus_state = StateVector([1, 1])
        register = Register([zero_state, plus_state], N=4)
        targets = [1]
        controls = [0]
        gate = ControlledGate(register, targets, controls)
        real_state = StateVector(zero_state.tensor(plus_state))
        self.assertTrue(gate.get_state() == real_state)
        
    def test_get_state_two_targets_one_control(self):
        zero_state = StateVector([1, 0])
        plus_state = StateVector([1, 1])
        register = Register([zero_state, plus_state], N=4)
        targets = [1, 2]
        controls = [3]
        gate = ControlledGate(register, targets, controls)
        real_state = StateVector(zero_state.tensor([plus_state, zero_state]))
        self.assertTrue(gate.get_state() == real_state)
        
    def test_get_state_one_target_two_control(self):
        zero_state = StateVector([1, 0])
        plus_state = StateVector([1, 1])
        register = Register([zero_state, plus_state], N=4)
        targets = [1]
        controls = [0, 3]
        gate = ControlledGate(register, targets, controls)
        real_state = StateVector(zero_state.tensor([zero_state, plus_state]))
        self.assertTrue(gate.get_state() == real_state)
        
    def test_get_state_two_target_two_control(self):
        zero_state = StateVector([1, 0])
        plus_state = StateVector([1, 1])
        register = Register([zero_state, plus_state], N=4)
        targets = [1, 2]
        controls = [0, 3]
        gate = ControlledGate(register, targets, controls)
        real_state = StateVector(zero_state.tensor([zero_state, plus_state, zero_state]))
        self.assertTrue(gate.get_state() == real_state)
        
    def test_to_qubit_one_target(self):
        one_state = StateVector([0, 1])
        zero_state = StateVector([1, 0])
        register = Register([zero_state, one_state, zero_state, zero_state])
        targets = [1]
        controls = [0]
        gate = ControlledGate(register, targets, controls, gate_name='x')
        real_states = [one_state]
        test_state = StateVector(zero_state.tensor(one_state))
        self.assertTrue(len(gate.to_qubit(test_state)) == 1)
        for state, real_state in zip(gate.to_qubit(test_state), real_states):   
            self.assertTrue(state == real_state)
        
    def test_to_qubit_two_targets(self):
        one_state = StateVector([0, 1])
        zero_state = StateVector([1, 0])
        register = Register([zero_state, one_state, zero_state, zero_state])
        targets = [1, 2]
        controls = [0]
        gate = ControlledGate(register, targets, controls, gate_name='swap')
        real_states = [one_state, zero_state]
        test_state = StateVector(zero_state.tensor([one_state, zero_state]))
        self.assertTrue(len(gate.to_qubit(test_state)) == 2)
        for state, real_state in zip(gate.to_qubit(test_state), real_states):   
            self.assertTrue(state == real_state)
        
class TestCNOTGate(TestMultiQubitGate):
    
    def test_matrix_rep(self):
        register = Register(N=2)
        targets = [1]
        controls = [0]
        gate = CNOT(register, targets, controls)
        real_gate = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        self.assertTrue(np.all(gate.matrix_rep() == real_gate))
        
    def test_evolve_control_zero(self):
        zero_state = StateVector([1, 0])
        one_state = StateVector([0, 1])
        register = Register([zero_state, one_state], N=4)
        targets = [1]
        controls = [0]
        gate = CNOT(register, targets, controls)
        gate.evolve()
        real_reg = Register([zero_state, one_state], N=4)
        for qubit, real_qubit in zip(register, real_reg):
            self.assertTrue(qubit == real_qubit)
            
    def test_evolve_control_one(self):
        zero_state = StateVector([1, 0])
        one_state = StateVector([0, 1])
        register = Register([one_state, zero_state], N=4)
        targets = [1]
        controls = [0]
        gate = CNOT(register, targets, controls)
        gate.evolve()
        real_reg = Register([one_state, one_state])
        for qubit, real_qubit in zip(register, real_reg):
            self.assertTrue(qubit == real_qubit)
            
    def test_evolve_control_plus(self):
        zero_state = StateVector([1, 0])
        one_state = StateVector([0, 1])
        plus_state = StateVector([1, 1])
        register = Register([plus_state, one_state], N=4)
        targets = [1]
        controls = [0]
        gate = CNOT(register, targets, controls)
        gate.evolve()
        real_reg = Register([plus_state, zero_state], N=4)
        for qubit, real_qubit in zip(register, real_reg):
            self.assertTrue(qubit == DensityMatrix(real_qubit.to_density_matrix()))
            
class TestCSWAPGate(TestMultiQubitGate):
    
    def test_matrix_rep(self):
        register = Register(N=2)
        targets = [0, 1]
        gate = CSWAP(register, targets)
        real_gate = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        self.assertTrue(np.all(gate.matrix_rep() == real_gate))
        
    def test_matrix_rep(self):
        register = Register(N=2)
        targets = [1]
        controls = [0]
        gate = CNOT(register, targets, controls)
        real_gate = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        self.assertTrue(np.all(gate.matrix_rep() == real_gate))
        
    def test_evolve_control_zero(self):
        zero_state = StateVector([1, 0])
        one_state = StateVector([0, 1])
        register = Register([zero_state, one_state], N=4)
        targets = [1, 2]
        controls = [0]
        gate = CSWAP(register, targets, controls)
        gate.evolve()
        real_reg = Register([zero_state, one_state, zero_state, zero_state])
        for qubit, real_qubit in zip(register, real_reg):
            self.assertTrue(qubit == real_qubit)
            
    def test_evolve_control_one(self):
        zero_state = StateVector([1, 0])
        one_state = StateVector([0, 1])
        register = Register([one_state, one_state], N=4)
        targets = [1, 2]
        controls = [0]
        gate = CSWAP(register, targets, controls)
        gate.evolve()
        real_reg = Register([one_state, zero_state, one_state, zero_state])
        for qubit, real_qubit in zip(register, real_reg):
            self.assertTrue(qubit == real_qubit)
            
    def test_evolve_control_plus(self):
        zero_state = StateVector([1, 0])
        one_state = StateVector([0, 1])
        plus_state = StateVector([1, 1])
        register = Register([plus_state, one_state], N=4)
        targets = [1, 2]
        controls = [0]
        gate = CSWAP(register, targets, controls)
        gate.evolve()
        real_reg = Register([plus_state, zero_state], N=4)
        for qubit, real_qubit in zip(register, real_reg):
            self.assertTrue(qubit == DensityMatrix(real_qubit.to_density_matrix()))
        

if __name__ == '__main__':
    unittest.main()