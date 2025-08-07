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

