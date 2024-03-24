import unittest
import numpy as np
from hamiltonian.hamiltonian import SimpleQuackHamiltonian
TOL = 1e-6

class TestQuackHamiltonian(unittest.TestCase):

    def test_energy(self):
        h = SimpleQuackHamiltonian()
        values = {"XX": 2, "YY": 3, "ZZ": -2}
        assert abs(h.get_energy(values) + 3.0) < TOL

    def test_ansatz_energy(self):
        h = SimpleQuackHamiltonian()
        values = {}
        ops = ["XX", "YY", "ZZ"]
        shots = 512
        theta = np.pi
        for op in ops:
            counts = {0: 0, 1: 0}
            for _ in range(shots):
                qc = h.construct_ansatz(theta=theta, op=op)
                qc.run_with_ops()
                result = qc.classical_storage[0]
                if np.array_equal(result, [[1, 0], [0, 0]]):
                    counts[0] += 1
                else:
                    counts[1] += 1
            print(counts)
            values[op] = (counts[0] - counts[1]) / shots
        print(values)
        print(h.get_energy(values))
        assert abs(values["XX"] - 1) < TOL
        assert abs(values["YY"] - 1) < TOL
        assert abs(values["ZZ"] + 1) < TOL
        assert abs(h.get_energy(values) + 1) < TOL
        



if __name__ == "__main__":
    unittest.main()