import math
import random

def IsPrime(p):
  If(p==1):
  return False

  if(p==2 or p==3 or p==5 or p==7):
    return True

  if(p%2==0):
    return False

  i=3
  while(i**2<p):
    if(p%i==0):
      return False
    i+=2

  return True


def PrimeGen(n,t):
  for i in range(t):
    p=random.randrange(2**(n-1)+1, 2**n)
    if IsPrime(p):
      return p
  return False


n=20
for k in range(5):
  print(PrimeGen(n,3*n**2))
#에라토네스 체를 이용한 소수 뽑기
