# QuACK
QuACK will be a highly optimized Variational Quantum Eigensolver (VQE) written in Python. 

## Background
The VQE is an algorithm that, given a Hamiltonian, finds its lowest eigenvalue and corresponding eigenvector [5]. It was first invented by [6]. One big motivation for the development of the VQE is quantum chemistry. A common problem in quantum chemistry is to find the minimal eigenvalue and its eigenvector of a molecule. However, the number of interactions between the electrons of a molecule scales exponentially with the number of electrons in that molecule [7], and so classical algorithms will likely scale exponentially in runtime/memory as well. Quantum computers could solve this issue. One of the main algorithms for finding the minimal eigenvalue of a Hamiltonian is the Quantum Phase Estimation (QPE) algorithm [7]. There are two issues with QPE:
1. It requires the minimal eigenvector to compute the minimal eigenvalue.
2. It requires massive amounts of qubits that are not feasible right now (in the NISQ era)

VQE aims to address these by taking an iterative approach. Instead of attempting to compute the eigenvalue exactly, it iteratively approximates it. Effectively, this lets us use a shallower quantum circuit and run multiple computations on that circuit. Thus, we have moved the cost away from number of qubits, and towards number of repetitions of the iterative cycle. There are two parts to the VQE [5]: 
1. Execution on QPU: a parameterized quantum circuit (PQC) prepares a quantum state and then samples outcomes to get the expectation value of a given Hamiltonian.
2. Execution on CPU: minimize the above expectation value.

## References
[1] Cheng-Lin Hong, Luis Colmenarez, Lexin Ding, Carlos L. Benavides-Riveros, and Christian Schilling. Quantum parallelized variational quantum eigensolvers for excited states, 2023. https://arxiv.org/abs/2306.11844
[2] J.R. Johansson, P.D. Nation, and Franco Nori. Qutip: An open-source python framework for the dynamics of open quantum systems. *Computer Physics Communications*, 183(8):1760–1772, August 2012. https://www.sciencedirect.com/science/article/abs/pii/S0010465512000835

[3] Stefano Mangini. Vqe from scratch. https://github.com/stfnmangini/VQE_from_scratch, 2021.

[4] Atsushi Matsuo. Variational quantum eigensolver and its applications. In Shigeru Yamashita and Tetsuo Yokoyama, editors, *Reversible Computation*, pages 22–41, Cham, 2021. Springer International Publishing. https://link.springer.com/chapter/10.1007/978-3-030-79837-6_2

[5] Alberto Peruzzo, Jarrod McClean, Peter Shadbolt, Man-Hong Yung, Xiao-Qi Zhou, Peter J. Love, Alan Aspuru-Guzik, and Jeremy L. O’Brien. A variational eigenvalue solver on a photonic quantum processor. *Nature Communications*, 5(1), July 2014. https://arxiv.org/abs/1304.3061

[6] Jules Tilly, Hongxiang Chen, Shuxiang Cao, Dario Picozzi, Kanav Setia, Ying Li, Edward Grant, Leonard Wossnig, Ivan Rungger, George H. Booth, and Jonathan Tennyson. The variational quantum eigensolver: A review of methods and best practices. *Physics Reports*, 986:1–128, November 2022. https://arxiv.org/abs/2111.05176

[7] Quest. https://github.com/QuEST-Kit/QuEST, 2018.
