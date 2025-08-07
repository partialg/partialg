# START OF LICENSE DECLARATION.
#
# CC BY-NC-ND 4.0 License
#
# (Attribution-NonCommercial-NoDerivatives 4.0 International)
#
# Copyright (c) 2025 Dennis Lima
#
# YOU ARE FREE TO share — copy and redistribute the material in any medium 
# or format. The licensor cannot revoke these freedoms as long as you follow the 
# license terms.
#
# UNDER THE FOLLOWING TERMS:
#     (i) Attribution — You must give appropriate credit, provide a link to the 
# license, and indicate if changes were made. You may do so in any reasonable 
# manner, but not in any way that suggests the licensor endorses you or your 
# use.
#     (ii) NonCommercial — You may not use the material for commercial purposes .
#     (iii) NoDerivatives — If you remix, transform, or build upon the material, you 
# may not distribute the modified material.
#     (iv) No additional restrictions — You may not apply legal terms or technological 
# measures that legally restrict others from doing anything the license permits.
#
# Notices:
#     (i) You do not have to comply with the license for elements of the material in the 
# public domain or where your use is permitted by an applicable exception or 
# limitation.
#     (ii) No warranties are given. The license may not give you all of the permissions 
# necessary for your intended use. For example, other rights such as publicity, 
# privacy, or moral rights may limit how you use the material.
#     (iii) View this license online at https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en.
#
# END OF LICENSE DECLARATION.

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
