''' ALL
pinv: for symbolic or dense partial inversion

'''

import numpy as np
from sympy import ImmutableMatrix, simplify

def pinv(M, *args):
    ''' Partial inversion algorithm
    M: numpy ndarray of floats or of sympy symbols.
    args: tuple of matrix indices. E.g.: (0,0), (1,2).
    # COMMENT: For ndarrays with more than 2 axes, only the first two are considered.
    '''
    Z = M.copy()
    for idx in args:
        i, k = idx
        new = []
        for r in range(Z.shape[0]):
            newrow = []
            for s in range(Z.shape[1]):
                Z_ = Z[i,k]**-1
                if s == k:                
                    if r == i:
                        newrow.append( Z_ )
                    else:
                        newrow.append(  Z[r,k]*Z_ )
                else:
                    if r == i:
                        newrow.append(  -Z_*Z[i,s] )
                    else:
                        newrow.append(  Z[r,s] - Z[r,k] * Z_ * Z[i,s] )
            new.append( newrow )
        Z = np.array(new).copy()
    #
    return ImmutableMatrix(new)


def inv(a, do_simplify=False):
    indices = [ (i,i) for i in range(a.shape[0]) ]
    if do_simplify == True:
        return simplify( pinv(a, *indices ) )
    else:
        return pinv(a, *indices )