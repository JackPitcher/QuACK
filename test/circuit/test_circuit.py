import sys
sys.path.append(r'c:/Users/jackp/QuACK')

import unittest
import numpy as np
from circuit import QuantumCircuit, MeasurementOp, GateOp
from qubits import Register, StateVector, DensityMatrix


class TestCircuit(unittest.TestCase):
        
    def test_teleportation_one(self):
        input_qubit = StateVector([0, 1])
        register = Register([input_qubit], N=3)
        h_0 = GateOp(name="h", register=register, targets=[1])
        cnot_0 = GateOp(name="cnot", register=register, targets=[2], controls=[1])
        cnot_1 = GateOp(name="cnot", register=register, targets=[1], controls=[0])
        h_1 = GateOp(name="h", register=register, targets=[0])
        measure_0 = MeasurementOp(target=1, classical_storage=0)
        measure_1 = MeasurementOp(target=0, classical_storage=1)
        cnot_2 = GateOp(name='cnot', register=register, targets=[2], controls=[1])
        cz_0 = GateOp(name="cz", register=register, targets=[2], controls=[0])
        measure_2 = MeasurementOp(target=2, classical_storage=2)
        ops = [h_0, cnot_0, cnot_1, h_1, measure_0, measure_1, cnot_2, cz_0, measure_2]
        circuit = QuantumCircuit(reg=register, operations={}, num_classical_stores=3, ops=ops)
        circuit.run()
        self.assertTrue(circuit.classical_storage[2] == 1)
        
    def test_teleportation_zero(self):
        input_qubit = StateVector([1, 0])
        register = Register([input_qubit], N=3)
        h_0 = GateOp(name="h", register=register, targets=[1])
        cnot_0 = GateOp(name="cnot", register=register, targets=[2], controls=[1])
        cnot_1 = GateOp(name="cnot", register=register, targets=[1], controls=[0])
        h_1 = GateOp(name="h", register=register, targets=[0])
        measure_0 = MeasurementOp(target=1, classical_storage=0)
        measure_1 = MeasurementOp(target=0, classical_storage=1)
        cnot_2 = GateOp(name='cnot', register=register, targets=[2], controls=[1])
        cz_0 = GateOp(name="cz", register=register, targets=[2], controls=[0])
        measure_2 = MeasurementOp(target=2, classical_storage=2)
        ops = [h_0, cnot_0, cnot_1, h_1, measure_0, measure_1, cnot_2, cz_0, measure_2]
        circuit = QuantumCircuit(reg=register, operations={}, num_classical_stores=3, ops=ops)
        circuit.run()
        self.assertTrue(circuit.classical_storage[2] == 0)
        
if __name__ == '__main__':
    unittest.main()
    