import numpy as np
from experiment.experiment import QuackExperiment, QiskitExperiment, QutipExperiment
from optimizer.optimizer import GradientDescent, Adam
from hamiltonian.hamiltonian import SimpleQuackHamiltonian
from hamiltonian.hamiltonian_other import SimpleQiskitHamiltonian, SimpleQutipHamiltonian
from circuit_simulator.CircuitSimulator import NumbaSimulator, CUDASimulator
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_classical_counts(results):
    result = {0: 0, 1: 0}
    for shot in results:
        result[shot[0]] += 1
    return result


def find_optimal_shots():
    hamiltonian = SimpleQuackHamiltonian()
    simulator = NumbaSimulator
    shots = np.arange(1, 1e4, step=10)
    expected = -1.0
    results = []
    for shot in shots:
        vqe_res = {}
        for op in hamiltonian.get_ops():
            qc = hamiltonian.construct_ansatz(theta=np.pi, op=op)
            sim = simulator(qc, shot, "")
            sim.run()
            counts = get_classical_counts(sim.cs_result)
            vqe_res[op] = (counts[0] - counts[1])/shot
    
        energy = hamiltonian.get_energy(vqe_res)
        results.append(abs(energy - expected))
        print(shot, abs(energy))

    fig, ax = plt.subplots(1, 1)
    ax.plot(shots, results)
    plt.savefig("output/shots.pdf")

#find_optimal_shots()

hamiltonian = SimpleQuackHamiltonian()
optimizer = Adam()
sim_method = NumbaSimulator
experiment = QuackExperiment(hamiltonian, optimizer, sim_method)
experiment.set_param("shots", 32)

step_sizes = np.linspace(0.001, 1., num=10)
beta1s = np.linspace(0.85, 0.99, num=10, endpoint=False)
beta2s = np.linspace(0.99, 0.9999, num=10, endpoint=False)
num_iters =[1e2, 1e3]
num_attempts = 5

lowest_diff = np.inf
best_params = []

for num_iter in num_iters:
    print(f"Num Iter = {num_iter}")
    optimizer.set_param("Max Iterations", num_iter)
    for step_size in step_sizes:
        optimizer.set_param("Step Size", step_size)
        for beta1 in beta1s:
            optimizer.set_param("Beta 1", beta1)
            for beta2 in beta2s:
                optimizer.set_param("Beta 2", beta2)
                params = [num_iter, step_size, beta1, beta2]
                diffs = 0.
                for _ in range(num_attempts):
                    diffs += abs(experiment.run([3.1]) - np.pi)
                if diffs/num_attempts < lowest_diff:
                    lowest_diff = diffs/num_attempts
                    best_params = params
                print(f"Params: {params}, diff: {diffs/num_attempts}")

print(f"Lowest diff was {lowest_diff} with params {best_params}")