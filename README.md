# partialg (v0.1)

**Motivation**

Partial algebra methods are partial implementations of linear algebraic operations for n-dimensional arrays. Their use cases include matrix compression, parallelization of matrix operations, approximate eigensolving, exact symbolic matrix inversion, generalized rectangular matrix inversion, definition of properties of zero-base polynomials (extending on Narayana's "kha-guna" from https://www.jstor.org/stable/224869), isomorphism between pseudo-unitary groups. Potential impact areas include Pseudo-Unitary Quantum Mechanics, Spectral Theory (Linear Algebra), Data Analytics, Machine Learning, Molecular Simulation (Hamiltonian compression).

---
**About**

This package provides a partial inversion function and a Sridhara-based block-diagonalization function in python (version 3.12.0).

Requirements: numpy, scipy, sympy.

Requirements for examples: matplotlib, pennylane, jax, jaxlib, optax

(!) Warning: jax and jaxlib may show version compatibility issues with pennylane if the correct versions are not met.

---
**How to Cite**
- For applications and modifications the Partial Inversion algorithm, cite this paper:
  
  **Dennis Lima and Saif Al-Kuwari. Unitarization of pseudo-unitary quantum circuits in the S-matrix framework. 2024 Phys. Scr. 99 045202. URL: https://doi.org/10.1088/1402-4896/ad298a**

- For applications and modifications to the Sridhara-based Block Diagonalization algorithm, cite this paper:

  **Dennis Lima and Saif Al-Kuwari. Sridhara-Compressed VQE Accelerates Molecular Energy Ranking of Polyaromatic Hydrocarbons. 2025 quant-ph arXiv. URL: https://arXiv.org/abs/2507.12678**

---
**How to Use**
1. Download the partialg folder, extract it and change your working directory to the parent directory of the partialg folder.
2. Open a python environment (supported version: 3.12.0).
3. Import the functions using **from partialg.partial_inversion import pinv** and **from partialg.eigensolvers import sbd_eigvals**.
4. See the usage examples in the .py files for guidance.

---
**Related pages from the web**

Partial inversion on MathOverflow: https://mathoverflow.net/questions/186026/partial-inverse-of-a-matrix-or-does-it-have-its-own-name/477652#477652

Partial inversion on Wikipedia: https://en.wikipedia.org/wiki/Partial_inverse_of_a_matrix

Cloning and deleting in a pseudo-unitary system: https://link.springer.com/article/10.1007/s11467-021-1063-z

Properties of block matrices: https://en.wikipedia.org/wiki/Block_matrix

---
**DISCLAIMERS**

This repository is updated for bug fixes, examples and descriptions on a monthly basis.


