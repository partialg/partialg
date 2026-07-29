#----------------------------------------
# This submodule defines special symbols for id consistency in
# expressions -- required for substitutions in sympy.
#
# EXAMPLES of special conjugations defined here:
# mconjugate( khadva ) = 1j*khadva
# sconjugate( oo ) = -oo
#----------------------------------------

from sympy import symbols, sqrt
o      = symbols('o')  # khaguna
oo     = 1/o           # khahara

khaguna = o             # same as khaguna above
khahara = oo            # same as khahara above
khadva  = sqrt(o - oo)  # khadva

def mconjugate(expr):
    "Memory-space conjugate mapping khagunas to khaharas by inversion."
    return expr.subs(o, oo)

def sconjugate(expr):
    "Memory-space sign conjugate flipping the signs of khagunas."
    return expr.subs(o, -o)



__all__ = (o, oo, khaguna, khahara, khadva, mconjugate, sconjugate)
