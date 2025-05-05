#카라츠바 알고리즘
import math

a=411
b=52121

def Karat(a,b):
  if len(str(a)) == 1 or len(str(b)) == 1:
    return a*b

  else:
    n2 = max( len(str(a)), len(str(b) ))
    n = n2 // 2

    a1=  a // (10**n)
    a0= a % (10**n)

    b1= b // (10**n)
    b0= b % (10**n)

    r0 = Karat(a0,b0)
    r1 = Karat(a0+a1,b0+b1)
    r2 = Karat(a1,b1)

    return r0 + ((r1-r2-r0)*(10**n)) + r2* (10**(2*n))

c = Karat(a,b)
print(c,a*b)
