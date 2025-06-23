# partial_algebra

**Motivation**

Partial algebra methods are partial implementations of non-linear algebraic operations for n-dimensional arrays. Their use cases include matrix compression, parallelization of matrix operations, approximate eigensolving, exact symbolic matrix inversion, generalized rectangular matrix inversion, definition of properties of zero-base polynomials (extending on Narayana's "kha-guna" from jstor.org/stable/224869), isomorphism between pseudo-unitary groups. Potential impact areas include Pseudo-Unitary Physics, Spectral Theory (Linear Algebra), Data Analytics, Machine Learning.

---
**About**

This package includes a partial inversion routine and a Sridhara-based block-diagonalization routine in python (version 3.12.0).

---
**How to Cite**
- For the partial inversion algorithm, cite this paper:
  
  **Dennis Lima and Saif Al-Kuwari 2024 Phys. Scr. 99 045202. URL: https://doi.org/10.1088/1402-4896/ad298a**

- For the Sridhara-based Block-Diagonalization algorithm, cite this paper:

  **[Upcoming]**

---
**How to Use**
1. Download the partial_algebra folder, extract it and change your working directory to the parent directory of the partial_algebra folder.
2. Open a python (supported version: 3.12.0) environment.
3. Import the functions using **from partial_algebra.inversion import partial_inversion** and **from partial_algebra.eigensolvers import sbd_eigvals**.
4. See the usage examples in the .py files for guidance.
