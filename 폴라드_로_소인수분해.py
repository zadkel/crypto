import math #pollard Rho 소인수분해
import sympy

def F(x,N):
  return (x*x+1)%N

def PollardRho(N):
  y1=2
  y2=2
  d=1

  while(d==1):

    y1=F(y1)
    y2=F(F(y2))
    d = math.gcd(y1-y2,N)

  if (d==N):
    return False
  else:
    return d


def PolRho(N,x,y):
  n = math.ceil(math.ceil(math.log2(N)/2))
  n2 = math.ceil(pow(2,n/2)) #생일공격의 원리에 따라 sqrtN번만 뽑으면 충돌확률 50%넘음

  print("i,x,y,d")
  print("--------------")
  for i in range(n2):
    x=F(x,N)
    y=(F(F(x,N),N))
    d=math.gcd(x-y,N)
    print(i+1,x,y,d)

    if(1<d<N):
      return d

  print("fail")

if __name__=='__main__':
  N=41*43
  PolRho(N,2,2)

