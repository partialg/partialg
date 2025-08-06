from jax.numpy import array

def pinvs(M, *args):
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
        Z = array(new).copy()
    #
    return array(new)