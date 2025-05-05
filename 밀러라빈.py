import math,random #밀러라빈 소수 판별법

def MillerRabin(x,p):
  if( p%2==0):
    return False

  for i in range(2,int(math.log2(p)+1)): #퍼펙트 파워라면 탈락
    if(float(p**(1/i)).is_integer()):
      return False

  u=p-1
  r=0
  while(u%2==0):
    u=u//2
    r+=1

  if((p-1) != (u*2**r)):
    print("ERROR")

  xpow=pow(x,u,p)

  for i in range(r): #x^(u 2^d) 승이 +1, -1이 나오면 잠정소수로 보고 종료
    if(xpow ==1 or xpow == p-1):
      return True
    xpow = xpow**2

  return False #다 통과했는데도 +1,-1이 안나왔다고? 너 탈락


def PrimeGen(n):
    p=random.randrange(2**(n-1)+1, 2**n)
    if MillerRabin(2,p):
      return p, True
    else:
      return p, False


for i in range(30):
  print(PrimeGen(10))
