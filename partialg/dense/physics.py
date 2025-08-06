

from jax.numpy import array, where, kron, concatenate, sqrt

def odious_series(n):
    """ Returns sequence from first to nth odious number.
    """
    #
    z = array([1,-1]) 
    s = z.copy()
    w = array([1], dtype=int)
    for i in range(n):
        s = kron(s,z)
        w = where( s == -1 )[0]
        if w.shape[0] >= n:
            break
    #
    return w[:n]

def evil_series(n):
    """ Returns sequence from first to nth odious number.
    """
    #
    z_ = array([1,1j]) 
    s = z_.copy()
    w = array([0], dtype=int)
    for i in range(n):
        s = kron(s,z_)
        w = where( s == -1 )[0]
        if w.shape[0] >= n:
            break
    #
    return concatenate( (np.array([0]), w[:n-1]) )

def zpu_h():
    "Z-pseudo-unitary Hadamard quantum gate"
    return array([[sqrt(2),-1],[1,-sqrt(2)]])

def zpu_x( kha=0.0001):
    "Z-pseudo-unitary X quantum gate, with khaguna set numerically."
    return array([[-1,1],[-1,1]])/kha

def zpu_y(kha=0.0001):
    "Z-pseudo-unitary Y quantum gate, with khaguna set numerically."
    return array([[-1, -1j],[-1j, 1]])/kha

def zpu_z(kha=0.0001):
    "Z-pseudo-unitary Z quantum gate."
    return array([[1,kha],[kha,-1]])

def zpu_i(kha=0.0001):
    "Identity quantum gate."
    return array([[1,kha],[kha,1]])

def kha_gate(kha=0.0001):
    "Null quantum gate with khaguna basis."
    return array([[kha,kha],[kha,kha]])

def zpu_o(kha=0.0001):
    "Null quantum gate."
    return array([[kha,1],[-1,1/kha]])
