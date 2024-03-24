import numpy as np
from experiment.experiment import QuackExperiment, QiskitExperiment, QutipExperiment
from optimizer.optimizer import GradientDescent
from hamiltonian.hamiltonian import SimpleQuackHamiltonian
from hamiltonian.hamiltonian_other import SimpleQiskitHamiltonian, SimpleQutipHamiltonian

module = 'quack'

if module == "quack":
    hamiltonian = SimpleQuackHamiltonian()
    schedule = [[128, 256, 512], [0.1, 0.05, 0.01]]
    step_size = np.pi/1e2
    optimizer = GradientDescent(schedule=schedule, step_size=step_size)
    experiment = QuackExperiment(hamiltonian, optimizer)
    experiment.set_param("shots", 128)
    experiment.run([3.1], verbose=True)
elif module == 'qutip':
    hamiltonian = SimpleQutipHamiltonian()
    schedule = [[16, 64, 128], [0.1, 0.05, 0.01]]
    step_size = np.pi/1e2
    optimizer = GradientDescent(schedule=schedule, step_size=step_size)
    experiment = QutipExperiment(hamiltonian, optimizer)
    experiment.set_param("shots", 16)
    experiment.run([3.1], verbose=True)
elif module == 'qiskit':
    hamiltonian = SimpleQiskitHamiltonian()
    schedule = [[512, 1024, 2048], [0.1, 0.05, 0.01]]
    step_size = np.pi/1e2
    optimizer = GradientDescent(schedule=schedule, step_size=step_size)
    
    experiment = QiskitExperiment(hamiltonian, optimizer)
    experiment.set_param("shots", 1024)
    
    experiment.run([3.1], verbose=True)