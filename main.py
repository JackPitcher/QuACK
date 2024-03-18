import numpy as np
from experiment.experiment import Experiment
from optimizer.optimizer import scalar_minimizer, gradient_descent
from hamiltonian.hamiltonian_qiskit import SimpleQiskitHamiltonian
from quantumcircuitbuilder.qcb_qiskit import SimpleQiskitCircuit
from hamiltonian.hamiltonian_qutip import SimpleQutipHamiltonian
from quantumcircuitbuilder.qcb_qutip import SimpleQutipCircuit

module = 'qiskit'

if module == 'qutip':
    hamiltonian = SimpleQutipHamiltonian()
    qcb = SimpleQutipCircuit()
    optimizer = lambda x, y: scalar_minimizer(x, y, step_size=0.01, bs=(0, np.pi))
    experiment = Experiment(hamiltonian, qcb, optimizer, 'qutip')
    experiment.set_param("shots", 16)
    experiment.set_param("solver_iterations", 64)
    
    experiment.run([3.0], verbose=True)

elif module == 'qiskit':
    hamiltonian = SimpleQiskitHamiltonian()
    qcb = SimpleQiskitCircuit()
    #optimizer = lambda x, y: scalar_minimizer(x, y, step_size=0.005)
    #optimizer = lambda x, y: scalar_minimizer(x, y, step_size=0.01, bs=(0, np.pi))
    optimizer = lambda x, y: gradient_descent(x, y)
    experiment = Experiment(hamiltonian, qcb, optimizer, 'qiskit')
    experiment.set_param("shots", 1024)
    experiment.set_param("solver_iterations", 2048)
    
    experiment.run([3.1], verbose=True)