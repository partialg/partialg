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



from time import perf_counter           # For time measurement 

from numpy.linalg import inv, eig
from jax.numpy import eye, sqrt, array_split, array, log2, diag
from jax.numpy import abs as npabs
#from numpy import eye, sqrt, array_split, array, log2, diag
#from numpy import abs as npabs

from scipy.sparse import coo_array
from scipy.sparse.linalg import eigs



# def exact_sqrt(a):
#      """Eigensolver way to compute matrix square roots. Not available for sparse matrices.
#      Availed for comparison purpose only. Not needed in the main algorithm.
#      """
#      e, v = eig( a )
#      e    = array(e, dtype=complex )
#      return v.dot( diag( sqrt( e ) ).dot( v.inv()) )


def ns_sqrt(a, max_it = 6, k_pow = 1/4):
    "Newton-Schulz matrix root expansion."
    A     = a.trace()**k_pow * eye(a.shape[0])   # Initial guess
    for i in range(max_it):
        A = 0.5*(A + a @ inv(A) )
    return A


# Slice blocks of matrix =====================
def block(a, nrow=2):
    ''' Splits matrix M into nrow*nrow blocks. Blocks have equal size if len(M)/nrow is integer.
    #
    INPUT  <np.array> : sparse matrix not allowed.
    OUTPUT <tuple(np.array)>
    '''
    #
    rows   = array_split(a, indices_or_sections=nrow, axis=0 ) 
    #
    blocks = []
    for row in rows:
        blocks.append( 
            array_split(row, indices_or_sections=nrow, axis=1 )
        )
    #
    return tuple(blocks)


#==============================================

def sbd_eigenvalue(a, sqrt= ns_sqrt):
    ''' Matrix-polynomial root via Sridhara-based Block Diagonalization method.
    PARAMETERS
        a            : matrix to take block-Bhaskara of. Accepts np.array or scipy sparse array.
        srt <np.array>: function to compute matrix square root
    OUTPUT
        <np.array>
    '''
    blk       = block(a, nrow=2)
    A, B      = blk[0][0], blk[0][1]
    C, D      = blk[1][0], blk[1][1]
    #
    del blk
    #
    t = A + D        # Block-trace    
    #
    try:             # Block-determinant with inverse of A
        A_ = inv(A)
        d  = A.dot(D) - A.dot(C.dot( A_.dot(B) ))
    except:          # Without inverse of A
        print('NOTE: Used singular matrix method.')
        d  = A.dot(D) - C.dot(B)
    #
    term = sqrt( t.dot(t) - 4*d )
    L0   = 0.5*(t - term)
    L1   = 0.5*(t + term)
    #
    return (L0, L1)


def sbd_vector(v, normalize=False):
    ''' Sridhara-based Block Diagonalization compressor for vectors
    INPUTS
        v <array-like>  : numpy dense 2D array or scipy sparse 2D array with shape (n,1) for any integer n>0.
        normalize <bool>: if True, normalizes compressed vector to recover unitarity, otherwise returns raw compressed vector.
        sparse <bool>   : if True, uses sparse methods, otherwise uses dense array methods.
    OUTPUT
        eigenvalue : if close to 1, compression had good quality.
        eigenvector: array-like of shape (n/2,2) where (n,2) is the shape of the input vector v.
    NOTES
        Can be used with any vector, but will only search for eigenvector 
            whose compressed density matrix eigenvalue (given vector v) is close to 1.
        Can only compress vector with even dimension.
        Intended for use with unitary vectors, especially for state 
            compression in quantum computing for quantum chemistry.
        If using with not unitary vector, it's advised to normalize your 
            vector before compression, using v = v/np.abs(np.sqrt((v.T.conjugate().dot(v))[0,0])) .
        May fail due to poor invertibility of blocks of density matrix.
    '''
    #
    if v.shape[0] %2 != 0:
        raise Warning('Shape is not even. Returning None.')
        return None
    #
    M   = v.dot(v.T.conjugate())
    L1  = sbd_eigenvalue(M)[1]
    L1  = coo_array( L1 )        
    #
    e, v = eigs( L1, k=1, sigma=1 )
    #
    if normalize == True:
        v = v/npabs( sqrt( v.T.conjugate().dot( v ) ) )
    #
    return e, array( v )

def sbd_vectorbranch(v, block_index='0', only_even=False, normalize=False ):
    ''' SBD_vectoreigbranch applies SBD_vector successively.
    block_index <int>: index of block-diagonal matrix (its length is the number of compressions).
    only_even <bool>: True ensures output only has elements with 2*n compressions, where n is the list index, as required by some VQE algorithms. 
                      False ensures output is full branch of compressed matrices.
    '''
    #
    t0 = perf_counter()
    #
    if 2**(len( block_index )-1) < v.shape[0]:
        L = [v, ]
        t = [0, ]
        for i in range( len(block_index) ):
            L.append( sbd_vector(L[-1], normalize=normalize)[ 1 ] )    # Block-eigensolving
            t.append( (perf_counter()-t0)/60. )
        #
        if only_even == True:
            L = [L[i] for i in range(0,len(L),2)]
            t = [t[i] for i in range(0, len(t), 2)]
    else:
        print(f'ABORTED: block_index is {int( len( block_index ) - log2(v.shape[0]) )  } indices too large.')
        L = None
        #
    report = {'time':t}    # Time is in minutes
    return L, report



def sbd_eigenbranch(M, block_index='0', only_even=False ):
    ''' SBD_eigbranch finds eigenleaf of block-eigenvalue tree.
    block_index <int>: index of block-diagonal matrix (its length is the number of compressions).
    only_even <bool>: True ensures output only has elements with 2*n compressions, where n is the list index, as required by some VQE algorithms. 
                      False ensures output is full branch of compressed matrices.
    '''
    #
    t0 = perf_counter()
    
    if 2**(len( block_index )-1) < M.shape[0]:
        L = [M, ]
        t = [0, ]
        for i in range( len(block_index) ):
            L.append( sbd_eigenvalue(L[-1])[ int(block_index[i]) ] )    # Block-eigensolving
            t.append( (perf_counter()-t0)/60. )
        #
        if only_even == True:
            L = [L[i] for i in range(0,len(L),2)]
            t = [t[i] for i in range(0, len(t), 2)]
    else:
        print(f'ABORTED: block_index is {int( len( block_index ) - log2(len(M)) )  } indices too large.')
        L = None
        #
    report = {'time':t}    # Time is in minutes
    return L, report


def sbd_eigenleaf(M, block_index='0'):
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
            L.append( sbd_eigenvalue(L[-1])[ int(block_index[i]) ] )    # Block-eigensolving
            t.append( (perf_counter()-t0)/60. )
            del L[0]
        #
    else:
        print(f'ABORTED: block_index is {int( len( block_index ) - log2(len(M)) )  } indices too large.')
        L = None
        #
    report = {'time':t}    # Time is in minutes

    return L[0], report

