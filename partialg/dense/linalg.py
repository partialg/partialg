# Provides matrix square root iterations


def lu_sqrt(a, is_hermitian=False):
    "Newton-Schulz matrix root expansion."
    e, v = np_eig(a)
    if is_hermitian == False:
        return v @ diag( sqrt(e) ) @ inv(v)
    else:
        return v @ diag( sqrt(e) ) @ v.T.conjugate()


def ns_sqrt(a: array, max_it : int = 9, k_pow : float = 1/4, convergence_threshold=None):
    "Newton-Schulz matrix root expansion."
    A     = a.trace()**k_pow * eye(a.shape[0])   # Initial guess
    median_convergence = []
    #
    if convergence_threshold != None:
        for i in range( max_it ):
            A_new = 0.5*(A + a @ inv(A) )
            median_convergence.append( np_max( np_abs( A_new - A ) ) )
            A = A_new.copy()
            #
            # Break loop if converged
            if median_convergence[-1] < convergence_threshold:
                print(f"GOOD: Converged at {i}th with maximum absolute error of {median_convergence[-1]}.") 
                break
            #
            if i == max_it-1:
                print(f"BAD: Didn't converge after {i}th with maximum absolute error of {median_convergence[-1]}.") 
                break
            #
            del A_new
    else:
        for i in range( max_it ):
            A = 0.5*(A + a @ inv(A) )
    #
    return A, median_convergence

def n_sqrt(a : array, max_it : int=20, convergence_threshold=None):
    "Newton iteration to approximate matrix square root."
    K                  = a.copy()
    median_convergence = []
    #
    if convergence_threshold != None:
        for i in range(max_it):
            K_new = (1/2)* ( K + inv(K) @ K )
            median_convergence.append( np_max( np_abs( K_new - K ) ) )
            K = K_new.copy()
            #
            # Breaking loop if converged
            if median_convergence[-1] < convergence_threshold:
                print(f"GOOD: Converged at {i}th with maximum absolute error of {median_convergence[-1]}.") 
                break
            #
            if i == max_it-1:
                print(f"BAD: Didn't converge after {i}th with maximum absolute error of {median_convergence[-1]}.") 
                break
            #
            del K_new
    else:
        for i in range(max_it):
            K = (1/2)* ( K + inv(K) @ K )
    #
    return K, median_convergence


# Slice blocks of matrix =====================
def block(a, nrow : int = 2):
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

def commutator(a,b):
    return a @ b - b @ a

def squared_norm(a):
    "Squared norm for complex matrices. Makes matrices Hermitian."
    return a @ a.T.conjugate()

def f_norm(a):
    "Frobenious norm of a matrix a"
    return np.sqrt( np.sum( a**2 ) )

def fl_norm(a):
    "Frobenious-Lima norm of commutators of blocks of a."
    A, B, C, D = block(a)
    #
    s0 = np.sum( squared_norm(commutator(A,B)) )
    s1 = np.sum( squared_norm(commutator(A,C)) )
    s2 = np.sum( squared_norm(commutator(A,D)) )
    s3 = np.sum( squared_norm(commutator(B,C)) )
    s4 = np.sum( squared_norm(commutator(B,D)) )
    s5 = np.sum( squared_norm(commutator(C,D)) )
    #
    return np.sqrt( (1/6)*(s0 + s1 + s2 + s3 + s4 + s5) )
