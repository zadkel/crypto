#Goal : generator g에 대해 g^x = h 일때, x를 찾기
#전제 g의 order가 q일 때,  q = q1*q2*q3... 소인수분해를 알고 있다고 가정

import math
import sympy

def PohligHellman(g,h,qfactors):
  clist=[]
  q=1
  for i in range(len(qfactors)): # q값 구하기
    q*=qfactors[i]

  for i in range(len(qfactors)): # qi, gi, hi 구하기
    qi = q//qfactors[i]
    gi = pow(g,qi,q+1)
    hi= pow(h,qi,q+1)
    xi = 0

    for k in range(q):
      if (pow(gi,k,q+1) == hi):
        xi = k
        break

    print(gi, "^",xi, "=", hi, "( mod" , q+1, "), xi = ", xi,"( mod",q//qi,")")

    clist.append(xi)


  x = invCRT(clist,qfactors,q)
  return x


def invCRT(clist,nlist,N):
  x=0
  Nlist=[]
  ulist=[]

  for n in nlist:
    Nhat = N//n
    Nlist.append(Nhat)
    u_i = pow(Nhat,-1,n)
    ulist.append(u_i)

  for ci,ui,Ni in zip(clist,ulist,Nlist):
    x= (x+ ci * ui* Ni) % N

  return x

if __name__ == '__main__':
  g=3
  h=11
  qfactors=[5,3,2]
  x=PohligHellman(g,h,qfactors)
  print("x =",x)
