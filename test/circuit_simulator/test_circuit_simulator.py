import sys
sys.path.append(r'c:/Users/jackp/QuACK')

from circuit_simulator import NumbaSimulator, CUDASimulator
from circuit import QuantumCircuit, GateOp, MeasurementOp
from qubits import Register, StateVector
import numpy as np
import unittest
TOL = 1e-6



def construct_ansatz(theta: float, op: str):
    register = Register([], N=2)
    ops = []
    ops.append(GateOp(name="h", register=register, targets=[0]))
    ops.append(GateOp(name="cnot", register=register, targets=[1], controls=[0]))
    ops.append(GateOp(name="rx", register=register, targets=[0], theta=theta))

    if op == "XX":
        ops.append(GateOp(name="h", register=register, targets=[0]))
        ops.append(GateOp(name="h", register=register, targets=[1]))
        ops.append(GateOp(name="cnot", register=register, targets=[1], controls=[0]))
        ops.append(MeasurementOp(target=1, classical_storage=0))
    elif op == "YY":
        ops.append(GateOp(name="sdg", register=register, targets=[0]))
        ops.append(GateOp(name="sdg", register=register, targets=[1]))
        ops.append(GateOp(name="h", register=register, targets=[0]))
        ops.append(GateOp(name="h", register=register, targets=[1]))
        ops.append(GateOp(name="cnot", register=register, targets=[1], controls=[0]))
        ops.append(MeasurementOp(target=1, classical_storage=0))
    elif op == "ZZ":
        ops.append(GateOp(name="cnot", register=register, targets=[1], controls=[0]))
        ops.append(MeasurementOp(target=1, classical_storage=0))
    elif op is not None:
        raise ValueError(f"Warning: Measurement on the {op} basis is not supported.")
        
    circuit = QuantumCircuit(reg=register, num_classical_stores=1, ops=ops)
    return circuit


def get_classical_counts(classical_storage: np.array, index: int):
    result = {0: 0, 1: 0}
    for shot in classical_storage:
        result[shot[index]] += 1
    return result



class TestNumbaSimulator(unittest.TestCase):
    num_shots = 1e3
    tol = 1e-6

    def test_simple_XX(self):
        circuit = construct_ansatz(np.pi, "XX")
        sim = NumbaSimulator(circuit, self.num_shots, "")
        sim.run()
        counts = get_classical_counts(sim.cs_result, 0)
        result = (counts[0] - counts[1]) / self.num_shots
        assert abs(result - 1.0) < self.tol

    def test_simple_YY(self):
        circuit = construct_ansatz(np.pi, "YY")
        sim = NumbaSimulator(circuit, self.num_shots, "")
        sim.run()
        counts = get_classical_counts(sim.cs_result, 0)
        result = (counts[0] - counts[1]) / self.num_shots
        assert abs(result - 1.0) < self.tol

    def test_simple_ZZ(self):
        circuit = construct_ansatz(np.pi, "ZZ")
        sim = NumbaSimulator(circuit, self.num_shots, "")
        sim.run()
        counts = get_classical_counts(sim.cs_result, 0)
        result = (counts[0] - counts[1]) / self.num_shots
        assert abs(result + 1.0) < self.tol

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
        circuit = QuantumCircuit(reg=register, num_classical_stores=3, ops=ops)
        sim = NumbaSimulator(circuit, self.num_shots, "")
        sim.run()
        for i in range(sim.cs_result.shape[0]):
            assert abs(sim.cs_result[i, 2] - 1) < self.tol
    
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
        circuit = QuantumCircuit(reg=register, num_classical_stores=3, ops=ops)
        sim = NumbaSimulator(circuit, self.num_shots, "")
        sim.run()
        for i in range(sim.cs_result.shape[0]):
            assert abs(sim.cs_result[i, 2] - 0) < self.tol


class TestCUDASimulator(unittest.TestCase):
    num_shots = 1e3
    tol = 1e-6

    def simple_XX(self):
        circuit = construct_ansatz(np.pi, "XX")
        sim = CUDASimulator(circuit, self.num_shots, "")
        sim.run()
        counts = get_classical_counts(sim.cs_result, 0)
        result = (counts[0] - counts[1]) / self.num_shots
        assert abs(result - 1.0) < self.tol

    def test_simple_YY(self):
        circuit = construct_ansatz(np.pi, "YY")
        sim = CUDASimulator(circuit, self.num_shots, "")
        sim.run()
        counts = get_classical_counts(sim.cs_result, 0)
        result = (counts[0] - counts[1]) / self.num_shots
        assert abs(result - 1.0) < self.tol

    def test_simple_ZZ(self):
        circuit = construct_ansatz(np.pi, "ZZ")
        sim = CUDASimulator(circuit, self.num_shots, "")
        sim.run()
        counts = get_classical_counts(sim.cs_result, 0)
        result = (counts[0] - counts[1]) / self.num_shots
        assert abs(result + 1.0) < self.tol

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
        circuit = QuantumCircuit(reg=register, num_classical_stores=3, ops=ops)
        sim = CUDASimulator(circuit, self.num_shots, "")
        sim.run()
        for i in range(sim.cs_result.shape[0]):
            assert abs(sim.cs_result[i, 2] - 1) < self.tol
    
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
        circuit = QuantumCircuit(reg=register, num_classical_stores=3, ops=ops)
        sim = CUDASimulator(circuit, self.num_shots, "")
        sim.run()
        for i in range(sim.cs_result.shape[0]):
            assert abs(sim.cs_result[i, 2] - 0) < self.tol
            

if __name__ == '__main__':
    unittest.main()