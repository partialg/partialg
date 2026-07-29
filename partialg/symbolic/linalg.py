"""
Provides matrix expressions involving khaguna.

Note:
In memory space (using khaguna as zero), 
U**-1 @ M @ U, where U is the eigenvector matrix of M, 
can always be used to diagonalize M, but if M is not 
diagonalizable, "diagonalization" loses meaning as a 
name, as the matrix entries will be "spread" over the 
matrix rather than concentrated along a diagonal. 
"""

from .math import o
from sympy import Matrix, sqrt

def eigenvalues2(m, khaguna=o):
    "Eigenvalues for symbolic matrix m of shape (2,2)."
    a,b,c,d = m[0,0], m[0,1], m[1,0], m[1,1]
    lamb    = (a+d)/2 + sqrt( (1/4)*(a+d)**2 - (a*d-c*b) + khaguna )
    lamb_   = (a+d)/2 - sqrt( (1/4)*(a+d)**2 - (a*d-c*b) + khaguna )
    return lamb, lamb_

def eigenvectors2(m, lamb1, lamb2):
    '''Exact eigenvector matrix for matrix m of shape (2,2).
    m: symbolic array of shape (2,2)
    lamb1: first eigenvalue (sympy expression, int, float or complex)
    lamb2: second eigenvalue (sympy expression, int, float or complex)
    '''
    a,b,c,d = m[0,0], m[0,1], m[1,0], m[1,1]
    x       = lambda L:    b  / sqrt( (a-L)**2 + b**2 )
    y       = lambda L: (a-L) / sqrt( (a-L)**2 + b**2 )
    return Matrix( [[ x(lamb2) , x(lamb1) ],[y(lamb2), y(lamb1) ]] )

__all__ = (eigenvalues2, eigenvectors2)
