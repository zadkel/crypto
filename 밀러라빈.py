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


'''
밀러라빈에 사용되는 정리.

만약 N이 홀수거나, 퍼펙트 스퀘어가 아닌 경우. Zn* 의 원소 중 적어도 절반은 강하게 합성수 테스트를 통과함. (즉, a^u != +-1, or a^2k u != -1 (mod N) for all i)

(x^16+1) = (x^8+1)(x^4+1)(x^2+1)(x+1)(x-1) =0 mod N 이런 식으로 인수분해 되니까

-> B = set of a s.t. a^2iu = +-1 B' = set of a s.t a^2iu = -1

그러면 B⊂B' 이때, B'는 Zn의 strrictly subgroup이므로 |B|<=|B'|<|Zn|/2
'''
