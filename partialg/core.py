
def pinv(M, *args, **kwargs):
    ''' Partial inversion algorithm
    M: numpy ndarray of floats or of sympy symbols.
    args: tuple of matrix indices. E.g.: (0,0), (1,2).
    # COMMENT: For ndarrays with more than 2 axes, only the first two are considered.
    '''
    mode = kwargs.get('mode', 'sparse')
    if mode == 'dense':
        from .dense.inversion import pinv
        return pinv(M, *args)
    elif mode == 'symbolic':
        # Yes, it's the same as for 'dense'
        from .dense.inversion import pinv 
        return pinv(M, *args)
    elif mode == 'sparse':
        raise Warning('ABORTED. Sparse partial inversion not currently supported.')
    else:
        raise Warning('ABORTED. Mode not supported.') 
    

def sbd(a, **kwargs):
    ''' Matrix-polynomial root via Sridhara-based Block Diagonalization method.
    PARAMETERS
        a            : matrix to take block-Bhaskara of. Accepts np.array or scipy sparse array.
        srt <np.array>: function to compute matrix square root
    OUTPUT
        <np.array>
    '''
    mode = kwargs.get('mode', 'sparse')
    #
    if mode == 'sparse':
        from .sparse.compression import sbd_eigenvalues
        return sbd_eigenvalues(a)
    elif mode == 'dense':
        from .dense.compression import sbd_eigenvalue
        return sbd_eigenvalue(a)
    elif mode == 'symbolic':
        from .symbolic.compression import sbd_eigenvaluey
        return sbd_eigenvaluey(a)
    else:
        raise Warning("ABORTED. Only sparse, dense or symbolic are supported.")

#