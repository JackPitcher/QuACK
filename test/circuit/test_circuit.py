import sys
sys.path.append(r'c:/Users/jackp/QuACK')

import unittest
import numpy as np
from circuit import QuantumCircuit
from qubits import Register, StateVector, DensityMatrix


class TestCircuit(unittest.TestCase):
    
    def test_teleportation(self):
        input_qubit = StateVector([1/np.sqrt(3), np.sqrt(2/3)])
        register = Register([input_qubit], N=3)
        operations = {
            "h_0": {
                "targets": [1],
                "controls": []
            },
            "cnot_0": {
                "targets": [2],
                "controls": [1]
            },
            "cnot_1": {
                "targets": [1],
                "controls": [0]
            },
            "h_1": {
                "targets": [0],
                "controls": []
            },
            "measure_0": {
                "target": 1,
                "classical_store": 0
            },
            "measure_1": {
                "target": 0,
                "classical_store": 1
            },
            "cnot_2": {
                "targets": [2],
                "controls": [1]
            },
            "cz_0": {
                "targets": [2],
                "controls": [0]
            }
        }
        circuit = QuantumCircuit(reg=register, operations=operations, num_classical_stores=2)
        circuit.run()
        final_register = circuit.reg
        self.assertTrue(final_register[2] == DensityMatrix(input_qubit.to_density_matrix()))
        
if __name__ == '__main__':
    unittest.main()
    