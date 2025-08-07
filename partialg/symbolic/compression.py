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

from time import perf_counter

from .inversion import invy
from sympy import eye
from sympy import factorint #Used to count number of factors of 2
from sympy import sqrt, det, simplify, expand
from sympy import symbols
from numpy import log2


K = symbols('K') # Coefficient used in the NS_sqrt function.

def ns_sqrty(a, max_it = 6, k_pow = 1/4, do_simplify=False):
    "Newton-Schulz matrix root expansion."
    A     = K * eye(a.shape[0])   # Initial guess
    for i in range(max_it):
        A = 0.5*(A + a * inv(A, do_simplify=do_simplify) )
    return A, K


# Slice blocks of matrix =====================

def sridhara_rooty(a, do_simplify=False):
    d = sqrt( a.trace()**2 - 4*det(a) )
    A = 0.5*(a.trace() - d)
    B = 0.5*(a.trace() + d)
    if do_simplify == True:
        return (simplify(A), simplify(B) )
    else:
        return (A, B)


def blocky(a, nrow=2):
    ''' Splits matrix M into nrow*nrow blocks. Blocks have equal size if len(M)/nrow is integer.
    #
    INPUT  <np.array> : sparse matrix not allowed.
    OUTPUT <tuple(np.array)>
    '''
    #
    m = []
    k = int(a.shape[0]/nrow )
    #
    for i in range(nrow):
        row = []
        for j in range(nrow):
            row.append( a[ i*k : (i+1)*k,  j*k : (j+1)*k ]  )
        #
        m.append(row)
    #
    return tuple(m) 


#==============================================

def sbd_eigenvaluey(a, sqrt= ns_sqrty, do_simplify=False, allsymbols={K}):
    ''' Matrix-polynomial root via Sridhara-based Block Diagonalization method.
    PARAMETERS
        a            : matrix to take block-Bhaskara of. Accepts np.array or scipy sparse array.
        srt <np.array>: function to compute matrix square root
    OUTPUT
        <np.array>
    '''
    blk       = blocky(a, nrow=2)
    A, C      = blk[0][0], blk[0][1]
    D, B      = blk[1][0], blk[1][1]
    #
    t = A + B        # Block-trace
    #
    try:             # Block-determinant with inverse of A
        A_ = inv(A)
        d  = A * B - A * D * A_ * C
    except:          # Without inverse of A
        print('Exception')
        d  = A * B - D * C
    #
    term, K = sqrt( t * t - 4*d )
    L0   = 0.5*(t - term)
    L1   = 0.5*(t + term)
    #
    if do_simplify == True:
        return (simplify(L0), simplify(L1) ), allsymbols
    else:
        return (L0, L1), allsymbols.union( {K})


def sbd_eigenbranchy(M, block_index='0', only_even=False, do_simplify=False, allsymbols={K}  ):
    ''' SBD_eigbranch finds eigenleaf of block-eigenvalue tree.
    block_index <int>: index of block-diagonal matrix (its length is the number of compressions).
    only_even <bool>: True ensures output only has elements with 2*n compressions, where n is the list index, as required by some VQE algorithms. 
                      False ensures output is full branch of compressed matrices.
    '''
    #
    t0 = perf_counter()
    #
    if 2**(len( block_index )-1) < M.shape[0]:
        L = [M, ]
        t = [0, ]
        for i in range( len(block_index) ):
            L0L1, allsymbols = sbd_eigenvaluey(L[-1], do_simplify=do_simplify, allsymbols=allsymbols)
            L.append( L0L1[ int(block_index[i]) ] )    # Block-eigensolving
            t.append( (perf_counter()-t0)/60. )
        #
        if only_even == True:
            L = [L[i] for i in range(0,len(L),2)]
            t = [t[i] for i in range(0, len(t), 2)]
    else:
        print(f'ABORTED: block_index is {int( len( block_index ) - log2(len(M)) )  } indices too large.')
        L = None
        #
    report = {'time':t, 'allsymbols':allsymbols}    # Time is in minutes
    return L, report


def sbd_eigenleafy(M, block_index='0', do_simplify=False, allsymbols={K}):
    ''' SBD_eigbranch finds eigenleaf of block-eigenvalue tree.
    Memory-economic SBD_eigenbranch, returning only the last block.
    '''
    #
    t0 = perf_counter()
    #
    if 2**(len( block_index )-1) < M.shape[0]:
        L = [M, ]
        t = [0, ]
        for i in range( len(block_index) ):
            L0L1, allsymbols = sbd_eigenvaluey(L[-1], do_simplify=do_simplify, allsymbols=allsymbols)
            L.append( L0L1[ int(block_index[i]) ] )    # Block-eigensolving
            t.append( (perf_counter()-t0)/60. )
            del L[0]
        #
    else:
        print(f'ABORTED: block_index is {int( len( block_index ) - log2(len(M)) )  } indices too large.')
        L = None
        #
    report = {'time':t, 'allsymbols':allsymbols}    # Time is in minutes
    return L[0], report

#
