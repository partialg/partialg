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


from numpy import array, where, kron, concatenate, sqrt

def nth_odious(i : int):
    return 2 * (i+1) - 1 - i.bit_count() % 2

def odious_series(start : int = 0, stop : int = None, step : int = 1):
    """ Returns sequence from (start) to (stop+1)th odious number at intervals of (step).
    """
    if stop == None:
        stop  = start
        start = 0
    #
    seq = []
    for i in range(start, stop, step):
        seq.append( nth_odious(i) )
    return tuple(seq)

def nth_evil(i : int):
    return 2 * (i+1) - 2 + (i.bit_count() % 2)

def evil_series(start : int = 0, stop : int = None, step : int = 1):
    """ Returns sequence from (start) to (stop+1)th evil number at intervals of (step).
    """
    if stop == None:
        stop  = start
        start = 0
    #
    seq = []
    for i in range(start, stop, step):
        seq.append( nth_evil(i) )
    return tuple(seq)

def h(**kwargs):
    "Z-pseudo-unitary Hadamard quantum gate"
    return array([[sqrt(2),-1],[1,-sqrt(2)]])

def x( kha=0.0001):
    "Z-pseudo-unitary X quantum gate, with khaguna set numerically."
    return array([[-1,1],[-1,1]])/kha

def y(kha=0.0001):
    "Z-pseudo-unitary Y quantum gate, with khaguna set numerically."
    return array([[-1, -1j],[-1j, 1]])/kha

def z(kha=0.0001):
    "Z-pseudo-unitary Z quantum gate."
    return array([[1,kha],[kha,-1]])

def i(kha=0.0001):
    "Identity quantum gate."
    return array([[1,kha],[kha,1]])

def kha_gate(kha=0.0001):
    "Null quantum gate with khaguna basis."
    return array([[kha,kha],[kha,kha]])

def o(kha=0.0001):
    "Z-pseudo-unitary Null quantum gate, with khaguna set numerically."
    return array([[kha,1],[-1,1/kha]])


