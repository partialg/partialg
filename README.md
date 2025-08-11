# PartiAlg (version 0.1)

CC BY-NC-ND 4.0 License

Copyright (c) 2025 Dennis Lima

### **About**
Provides partial implementations of linear algebraic operations for n-dimensional arrays (jax.numpy), sparse arrays (scipy) and symbolic matrices (sympy). Their use cases include matrix compression, parallelization of matrix operations, approximate eigensolving, exact symbolic matrix inversion, generalized rectangular matrix inversion, definition of properties of khaguna polynomials (<a href="https://www.jstor.org/stable/224869">read more</a>), isomorphism between pseudo-unitary groups. Potential impact areas include Pseudo-Unitary Quantum Mechanics, Spectral Theory (Linear Algebra), Data Analytics, Machine Learning, Molecular Simulation (Hamiltonian compression).

---
### 💬 **How to Cite**
- For applications and modifications of the Partial Inversion algorithm, cite this paper:
  
  **Dennis Lima and Saif Al-Kuwari. Unitarization of pseudo-unitary quantum circuits in the S-matrix framework. 2024 Phys. Scr. 99 045202. URL: https://doi.org/10.1088/1402-4896/ad298a**

- For applications and modifications of the Sridhara-based Block Diagonalization algorithm, cite this paper:

  [Submitted. Wait till publication.]

---
### 💻 **How to Use / Requirements**
Download and use:
1. Download the partialg folder, then unzip it.
2. Try the examples in the TUTORIAL file.


Installation from terminal using git clone from within a Jupyter Notebook:
`
!git clone https://github.com/partialg/partialg.git

# Add cloned path to your python path
import sys
sys.path.append('/<CLONED DIRECTORY>/partialg')    # If you're using Google Colab, your path will be '/content/partialg'

# Ready to use
import partialg
`

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
### 📚 **Topics related to this repo from the web**

- Partial inversion on MathOverflow (<a href="https://mathoverflow.net/questions/186026/partial-inverse-of-a-matrix-or-does-it-have-its-own-name/477652#477652">read more</a>).

- Partial inversion on Wikipedia (<a href="https://en.wikipedia.org/wiki/Partial_inverse_of_a_matrix">read more</a>)

- Cloning and deleting in a pseudo-unitary system (<a href="https://link.springer.com/article/10.1007/s11467-021-1063-z">read more</a>).

- Properties of block matrices (<a href="https://en.wikipedia.org/wiki/Block_matrix">read more</a>).

---
### 🥲 **Help me, I'm new to python...**

If you're new to python, follow these instructions.

WINDOWS:
1. Download python 3.11.9 and Microsoft Visual Studio from the Microsoft Store.
2. Download the partialg folder, then unzip it.
3. Type `pip list` in your terminal to ensure you have `pip` installed (it's a python package manager). If it's not installed, find a way to install it.
4. Open your terminal, paste and run the following command to install the dependencies:

`
pip install jax==0.4.28 jaxlib==0.4.28 numpy==2.0.2 matplotlib==3.9.2 scipy==1.16.1 sympy==1.13.3 tqdm==4.67.1 optax==0.2.5 pennylane==0.41.1
`


UBUNTU:

1. Run the lines below on terminal:

`sudo apt install python==3.11.9`

`sudo apt install python3-pip` or `sudo apt install python-pip`, whichever works first.

`
pip install jax==0.4.28 jaxlib==0.4.28 numpy==2.0.2 matplotlib==3.9.2 scipy==1.16.1 sympy==1.13.3 tqdm==4.67.1 optax==0.2.5 pennylane==0.41.1 notebook
`

2. Download the partialg folder, unzip it.
3. On terminal, change your current working directory to the same folder where the TUTORIAL folder is located (you can use the code below substituting "<YOUR DIRECTORY>" with the actual parent directory of your partialg folder). 

`cd <YOUR DIRECTORY>/partialg`

4. On terminal, open a Jupyter Notebook using the command below.

`jupyter notebook`

5. Use Jupyter Notebook interface to open the TUTORIAL notebook in the partialg folder and test the examples.

---
Sorted! 😊 Now consider the environment and make today your weekly vegan day 🌟.
