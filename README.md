# PartiAlg (version 0.0.1)

CC BY-NC-ND 4.0 License

Copyright (c) 2025 Dennis Lima

### **About**
Provides partial implementations of linear algebraic operations for n-dimensional arrays (numpy), sparse arrays (scipy) and symbolic matrices (sympy). Their use cases include matrix compression, parallelization of matrix operations, approximate eigensolving, exact symbolic matrix inversion, generalized rectangular matrix inversion, definition of properties of khaguna polynomials (<a href="https://www.jstor.org/stable/224869">read more</a>), isomorphism between pseudo-unitary groups. Potential impact areas include Pseudo-Unitary Quantum Mechanics, Spectral Theory (Linear Algebra), Data Analytics, Machine Learning, Molecular Simulation (Hamiltonian compression).

---
### 💬 **How to Cite**
- For applications and modifications of the Partial Inversion algorithm, cite this paper:
  
  **Dennis Lima and Saif Al-Kuwari. Unitarization of pseudo-unitary quantum circuits in the S-matrix framework. 2024 Phys. Scr. 99 045202. URL: https://doi.org/10.1088/1402-4896/ad298a**

- For applications and modifications of the Sridhara-based Block Diagonalization algorithm, cite this paper:
  
  **Dennis Lima and Saif Al-Kuwari. Sridhara-Compressed VQE Accelerates Molecular Energy Ranking of Polyaromatic Hydrocarbons. 2025 arXiv preprint arXiv:2507.12678. URL: https://arxiv.org/abs/2507.12678**

---
### 💻 **Installation and First Use**
1. Download from terminal (or jupyter notebook, using !pip):
```
pip install git+https://github.com/partialg/partialg.git
```
2. import functions and test them:
```
# NOTATION
# All function names are lowercase and singular.
# The suffixes -, -y, -s are for functions for dense, symbolic and sparse arrays, respectively.
# The prefix p- preceeds a standard name for partialized functions (inv -> pinv, eigval -> peigval).

# Proxies for dense, symbolic and sparse data types
from partialg import pinv    
from partialg import inv
from partialg import peigval

# Dense submodules
from partialg.inversion import pinv
from partialg.compression import peigval
from partialg.zpu_quantum import h, x, y, z, i.

# Other submodules
from partialg.sparse.compression import peigvals
from partialg.symbolic.compression import peigvaly

# Simple usage examples with a Hermitian matrix
import numpy as np
a       = np.random.rand(4,4) + 1j*np.random.rand(4,4)
a       = 0.5*( a + a.T.conjugate() )

a_00_01 = pinv(a, (0,0), (0,1) )
a_      = pinv(a, range(a.shape[0] ) )
b       = peigval(a)
```


Supported python packages for TUTORIAL:
- optax - 0.2.5
- pennylane - 0.41.1

---
### 📚 **Topics related to this repo from the web**

- Partial inversion on MathOverflow (<a href="https://mathoverflow.net/questions/186026/partial-inverse-of-a-matrix-or-does-it-have-its-own-name/477652#477652">read more</a>).

- Partial inversion on Wikipedia (<a href="https://en.wikipedia.org/wiki/Partial_inverse_of_a_matrix">read more</a>)

- Cloning and deleting in a pseudo-unitary system (<a href="https://link.springer.com/article/10.1007/s11467-021-1063-z">read more</a>).

- Properties of block matrices (<a href="https://en.wikipedia.org/wiki/Block_matrix">read more</a>).


---
Now consider the environment and make today your weekly vegan day 🌟.
