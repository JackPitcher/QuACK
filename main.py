import numpy as np
from experiment.experiment import QuackExperiment, QiskitExperiment, QutipExperiment
from optimizer.optimizer import GradientDescent
from hamiltonian.hamiltonian import SimpleQuackHamiltonian
from hamiltonian.hamiltonian_other import SimpleQiskitHamiltonian, SimpleQutipHamiltonian
import time
from tqdm import tqdm

module = 'quack'

N = 10

start = time.time()

for i in tqdm(range(N)):

    if module == "quack":
        hamiltonian = SimpleQuackHamiltonian()
        schedule = [[128, 256, 512], [0.1, 0.05, 0.01]]
        step_size = np.pi/1e2
        optimizer = GradientDescent(schedule=schedule, step_size=step_size)
        experiment = QuackExperiment(hamiltonian, optimizer)
        experiment.set_param("shots", 128)
        experiment.run([3.1], verbose=False)
    elif module == 'qutip':
        hamiltonian = SimpleQutipHamiltonian()
        schedule = [[128, 256, 512], [0.1, 0.05, 0.01]]
        step_size = np.pi/1e2
        optimizer = GradientDescent(schedule=schedule, step_size=step_size)
        experiment = QutipExperiment(hamiltonian, optimizer)
        experiment.set_param("shots", 128)
        experiment.run([3.1], verbose=False)
    elif module == 'qiskit':
        hamiltonian = SimpleQiskitHamiltonian()
        schedule = [[128, 256, 512], [0.1, 0.05, 0.01]]
        step_size = np.pi/1e2
        optimizer = GradientDescent(schedule=schedule, step_size=step_size)
        
        experiment = QiskitExperiment(hamiltonian, optimizer)
        experiment.set_param("shots", 128)
        
        experiment.run([3.1], verbose=False)
    
end = time.time()
print(f"AVERAGE TIME: {(end - start) / N}")