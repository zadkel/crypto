from fractions import Fraction

import sympy
from sympy.ntheory.continued_fraction import continued_fraction
from sympy.ntheory.continued_fraction import continued_fraction_reduce

e=10
N=7
c=Fraction(e,N)
print('e/n=',c)

cont_frac = continued_fraction(c)
print(cont_frac)

d=continued_fraction_reduce([1,2,3])
print(d)
