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
from scipy.sparse import eye
from scipy.sparse.linalg import inv
from scipy.sparse import csc_array #, csr_array
from scipy.sparse.linalg import eigs

from jax.numpy import sqrt, log2
from jax.numpy import abs as npabs

# def ExactSrt(a):
#     """Eigensolver way to compute matrix square roots. Not available for sparse matrices.
#     Availed for comparison purpose only. Not needed in the main algorithm.
#     """
#     e, v = np.linalg.eig( a )
#     e    = np.array(e, dtype=complex )
#     return v.dot( np.dot( np.diag( np.sqrt( e ) ), v.inv()) )


def ns_sqrts(a, max_it = 6, k_pow = 1/4):
    "Newton-Schulz matrix root expansion."
    A     = a.trace()**k_pow * eye(a.shape[0])   # Initial guess
    for i in range(max_it):
        A = 0.5*(A + a @ inv(A) )
    return A


# Slice blocks of matrix =====================
def blocks(a, nrow=2):
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
            row.append( csc_array( a[ i*k : (i+1)*k,  j*k : (j+1)*k ] ))
        #
        m.append(row)
    #
    return tuple(m) 


#==============================================

def sbd_eigenvalues(a, sqrt= ns_sqrts):
    ''' Matrix-polynomial root via Sridhara-based Block Diagonalization method.
    PARAMETERS
        a            : matrix to take block-Bhaskara of. Accepts np.array or scipy sparse array.
        srt <np.array>: function to compute matrix square root
    OUTPUT
        <np.array>
    '''
    blk       = blocks(a, nrow=2)
    A, C      = blk[0][0], blk[0][1]
    D, B      = blk[1][0], blk[1][1]
    #
    t = A + B        # Block-trace    
    #
    try:             # Block-determinant with inverse of A
        A_ = inv(A)
        d  = A.dot(B) - A.dot(D.dot( A_.dot(C) ))
    except:          # Without inverse of A
        print('Exception')
        d  = A.dot(B) - D.dot(C)
    #
    term = sqrt( t.dot(t) - 4*d )
    L0   = 0.5*(t - term)
    L1   = 0.5*(t + term)
    #
    return (L0, L1)


def sbd_vectors(v, normalize=False):
    ''' Sridhara-based Block Diagonalization compressor for vectors
    INPUTS
        v <array-like>  : numpy dense 2D array or scipy sparse 2D array with shape (n,1) for any integer n>0.
        normalize <bool>: if True, normalizes compressed vector to recover unitarity, otherwise returns raw compressed vector.
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
    L1  = sbd_eigenvalues(M)[1]      
    #
    e, v = eigs( L1, k=1, sigma=1 )
    #
    if normalize == True:
        v = v/npabs( sqrt( v.T.conjugate().dot( v ) ) )
    #
    return e, csc_array(v)


def sbd_vectorsbranch(v, block_index='0', only_even=False, normalize=False ):
    ''' sbd_vectorseigbranch applies sbd_vectors successively.
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
            L.append( sbd_vectors(L[-1], normalize=normalize)[ 1 ] )    # Block-eigensolving
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



def sbd_eigenbranchs(M, block_index='0', only_even=False ):
    ''' sbd_eigenbranchs finds eigenleaf of block-eigenvalue tree.
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
            L.append( sbd_eigenvalues(L[-1])[ int(block_index[i]) ] )    # Block-eigensolving
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



def sbd_eigenleafs(M, block_index='0'):
    ''' sbd_eigenbranchs finds eigenleaf of block-eigenvalue tree.
    Memory-economic SBD_eigenbranch, returning only the last block.
    '''
    #
    t0 = perf_counter()
    #
    if 2**(len( block_index )-1) < M.shape[0]:
        L = [M, ]
        t = [0, ]
        for i in range( len(block_index) ):
            L.append( sbd_eigenvalues(L[-1])[ int(block_index[i]) ] )    # Block-eigensolving
            t.append( (perf_counter()-t0)/60. )
            del L[0]
        #
    else:
        print(f'ABORTED: block_index is {int( len( block_index ) - log2(len(M)) )  } indices too large.')
        L = None
        #
    report = {'time':t}    # Time is in minutes
    return L[0], report
#

def transformed_eigs(M, T_factor=0, N_factor=1, make_Hermitian=True):
    ''' Finds ground state after multiplication of M by T_factor and sum by T_factor*eye(M.shape[0])
    '''
    #
    t0 = perf_counter()
    #
    if make_Hermitian == True:
        M2 = M @ M.T.conjugate()
        M2 = M2*N_factor + T_factor*eye(M2.shape[0])
        gs = sqrt( npabs((min( eigs( M2, sigma=0 )[0] ) -T_factor )/N_factor)  )
    else:
        M2 = M
        M2 = M2*N_factor + T_factor*eye(M2.shape[0])
        gs = (min( eigs( M2, sigma=0 )[0] ) -T_factor )/N_factor
    #    
    dt = perf_counter() - t0
    report = {'time':dt}    # Time is in minutes
    return gs, report

#
