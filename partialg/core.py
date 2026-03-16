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
from sympy import ImmutableMatrix

def pinv(M, *args, **kwargs):
    ''' Partial inversion algorithm
    M: np.array, scipy.sparse.csc_array or sympy.Matrix.
    args:   tuple of matrix indices. E.g.: (0,0), (1,2).
            If no args is given, returns M unchanged.
            If range is given as only args, generates pairs of equal indices from range.
    # COMMENT: For ndarrays with more than 2 axes, only the first two are considered.
    '''
    method = kwargs.get('method', 'sparse')
    #
    # Initialization from range
    if 'range' in str(type( args[0] )):
        args = tuple( [(i,i) for i in args[0]] )
    #
    # Methods
    if 'dense' == method:
        from .dense.inversion import inv
        return pinv(M, *args)
    elif 'symbolic' == method:
        # Yes, it's the same as for 'dense'
        from .dense.inversion import inv 
        return ImmutableMatrix( pinv(M, *args) )
    elif 'sparse' == method:
        raise Warning('ABORTED. Method not supported. Retuning None.')
        return None

def inv(a, **kwargs):
    "Full matrix inversion via partial inversion"
    return pinv(a, range(a.shape[0]))


def peigval(a, **kwargs):
    ''' Matrix-polynomial root via Sridhara Block Eigensolver method.
    PARAMETERS
        a            : 2D array to take block-Bhaskara of. Accepts np.array, scipy sparse array or sympy Matrix.
    '''
    method = kwargs.get('method', 'sparse')
    #
    if method == 'sparse':
        from .sparse.compression import peigvals
        return peigvals(a)
    elif method == 'dense':
        from .dense.compression import peigval
        return peigval(a)
    elif method == 'symbolic':
        from .symbolic.compression import peigvaly
        return peigvaly(a)
    else:
        raise Warning("ABORTED. Only sparse, dense or symbolic are supported.")


#
def help():
    "Prints example code"
    print('''# NOTATION
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
a_      = pinv(a, range(a.shape[0) ) )
b       = peigval(a)''')





