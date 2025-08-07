# PartiAlg (version 0.1)

CC BY-NC-ND 4.0 License

Copyright (c) 2025 Dennis Lima

**About**
Provides partial implementations of linear algebraic operations for n-dimensional arrays (jax.numpy), sparse arrays (scipy) and symbolic matrices (sympy). Their use cases include matrix compression, parallelization of matrix operations, approximate eigensolving, exact symbolic matrix inversion, generalized rectangular matrix inversion, definition of properties of khaguna polynomials ([https://www.jstor.org/stable/224869](read more)), isomorphism between pseudo-unitary groups. Potential impact areas include Pseudo-Unitary Quantum Mechanics, Spectral Theory (Linear Algebra), Data Analytics, Machine Learning, Molecular Simulation (Hamiltonian compression).

---
**How to Cite**
- For applications and modifications of the Partial Inversion algorithm, cite this paper:
  
  **Dennis Lima and Saif Al-Kuwari. Unitarization of pseudo-unitary quantum circuits in the S-matrix framework. 2024 Phys. Scr. 99 045202. URL: https://doi.org/10.1088/1402-4896/ad298a**

- For applications and modifications of the Sridhara-based Block Diagonalization algorithm, cite this paper:

  [Submitted. Wait till publication.]

---
**How to Use/Requirements**
For python users:
1. Download the partialg folder, then unzip it.
2. Try the examples in the TUTORIAL file.

You must have a compatible python version and compatible packages to avoid deprecation errors. Although in most cases newer versions will work, we have no warranty that partialg will work for versions different from the ones listed here.

Programming language:
- python - 3.11.9

Supported python packages (requirements):
- jax - 0.4.28
- jaxlib - 0.4.28
- matplotlib - 3.9.2
- numpy - 2.0.2
- scipy - 1.16.1
- sympy - 1.13.3
- tqdm - 4.67.1 

Supported python packages (for TUTORIAL only):
- optax - 0.2.5
- pennylane - 0.41.1

---
**Topics related to this repo from the web**

- Partial inversion on MathOverflow [https://mathoverflow.net/questions/186026/partial-inverse-of-a-matrix-or-does-it-have-its-own-name/477652#477652](link).

- Partial inversion on Wikipedia [https://en.wikipedia.org/wiki/Partial_inverse_of_a_matrix](link)

- Cloning and deleting in a pseudo-unitary system [https://link.springer.com/article/10.1007/s11467-021-1063-z](link).

- Properties of block matrices [https://en.wikipedia.org/wiki/Block_matrix](link).

---
**Help me, I never used python before!**
If you're brand new to python, follow these instructions.

WINDOWS:
1. Download python 3.11.9 and Microsoft Visual Studio from the Microsoft Store.
2. Download the partialg folder, then unzip it.
3. Type `pip list` in your terminal to ensure you have `pip` installed (it's a python package manager). If it's not installed, find a way to install it.
4. Open your terminal, paste and run the following command to install the dependencies:

`
pip install jax jaxlib numpy matplotlib numpy scipy sympy tqdm optax pennylane
`


UBUNTU:

1. Run the lines below on terminal:

`sudo apt install python==3.11.9`

`sudo apt install python3-pip`

`sudo apt install python3-pip`

`pip install jax jaxlib numpy matplotlib numpy scipy sympy tqdm notebook optax pennylane`

2. Download the partialg folder, unzip it and try the TUTORIAL jupyter notebook.

Done! 😊
