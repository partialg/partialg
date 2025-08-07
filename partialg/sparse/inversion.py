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

from jax.numpy import array

def pinvs(M, *args):
    ''' Partial inversion algorithm
    M: numpy ndarray of floats or of sympy symbols.
    args: tuple of matrix indices. E.g.: (0,0), (1,2).
    # COMMENT: For ndarrays with more than 2 axes, only the first two are considered.
    '''
    Z = M.copy()
    for idx in args:
        i, k = idx
        new = []
        for r in range(Z.shape[0]):
            newrow = []
            for s in range(Z.shape[1]):
                Z_ = Z[i,k]**-1
                if s == k:                
                    if r == i:
                        newrow.append( Z_ )
                    else:
                        newrow.append(  Z[r,k]*Z_ )
                else:
                    if r == i:
                        newrow.append(  -Z_*Z[i,s] )
                    else:
                        newrow.append(  Z[r,s] - Z[r,k] * Z_ * Z[i,s] )
            new.append( newrow )
        Z = array(new).copy()
    #

    return array(new)
