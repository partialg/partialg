
from compression import SBD_eigvals
import numpy as np
from matplotlib import pyplot as plt

def RandomMatrixTest(sample_size=100, T0=0, N0=1):
    ''' EXAMPLE USAGE OF SBD FUNCTION
    sample_size <int>: number of random matrices to test
    T0 <float>: factor to be added to initial matrix and subtracted from eigenvalues to improve quality.
    N0 <float>: factor to divide initial matrix and multiply eigenvalues to improve quality.

    '''
    for i in range(sample_size):
        try:
            M = np.random.rand(8,8)                # Generating random matrix M
            M = M.dot(M.conjugate().transpose() )  # Making M Hermitian
            M = M/M.trace()/N0 + T0*np.eye(8)                     # Normalizing M
            #
            L = SBD_eigvals(M)                     # Block-eigensolving M
            #
            # Eigensolving M and its blocks for comparison
            E1 = (np.sort( np.abs( list( np.linalg.eigvals(L[0]) ) + list( np.linalg.eigvals(L[1]) )  ) ) - T0 ) *N0 
            E2 = (np.sort( np.linalg.eigvals( M ) ) - T0) *N0
            #
            from matplotlib import pyplot as plt
            plt.bar(range(len(E1)), E1/E2, alpha=0.4, color='blue')
            #
            plt.ylim(0.1, 5)
            plt.xlim(-0.5, len(E1)-0.5)
            #
            plt.ylabel('Ratio of eigenvalues')
            plt.xlabel('Position of eigenvalue')
            plt.title('Ratio distribution for spectrum of (8,8)-sized random matrices')
            #
        except:
            continue
    #
    plt.hlines(y=1, xmin=-5, xmax= len(E1)+5, color='black', label='Best values')
    plt.legend()
    plt.show()

