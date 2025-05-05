import math

nlist = (5,3)
N=1
for n in nlist:
  N=N*n

print("N=",N)

def CRT(x, nlist):
  clist = []
  for n in nlist:
    clist.append( x%n )
  return clist


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



print (CRT(38,nlist))
print (invCRT([2,3,1],nlist,N))

#CRT의 중요한 점.  11^312 mod 15 같은 걸 계산할 때 작은 수로 쪼갠다음 계산하고 다시 합치는게 훨씬 빨라서.
