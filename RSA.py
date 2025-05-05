import math #RSA
import random

def is_perfect_power(N):
    for c in range(2,int(math.log2(N))+1):
        if (N**(1/c)).is_integer():
            return True
    return False

def MillerRabinTest(N, t):
    if (N == 2):
        return True
    if N % 2 == 0:  # N>2 is even ?
        return False

    # check if N is perfect power
    # if N**(1/c) is an integer, N is a perfect power

    if (is_perfect_power(N) == True):
        return False

    # Find u odd such that N-1 = 2^r * u
    u = N - 1
    r = 0
    while (u % 2 == 0):
        u //= 2
        r += 1

    for j in range(t):
        a = random.randint(1,N)
        b = pow(a,u,N) # b = a^u mod N

        if (b!=1) and (b!=N-1):
            for i in range (r): #(r-1)번 거듭제곱
                b = pow(b,2,N)
                if b == N-1:
                    #print(j, i, 'a^{2^i*u}')
                    break       # break and repeat with a new "a"
            return False        # a^{2^i*u}!=-1 (mod N) for all i
    return True

def gen_prime(n, t):
    t_MillerRabinTest = n
    for i in range(t):
        N = random.randrange(2**(n-1)+1, 2**n) # 2^{n-1}+1 <= p < 2^n
        if MillerRabinTest(N, t_MillerRabinTest):
            return N
    return False

def generate_keys(n_bit):

  pq_bit = n_bit//2
  t=3*pq_bit**2
  p = gen_prime(pq_bit,t)
  q = gen_prime(pq_bit,t)
  n=p*q
  return p, q, n

def enc(m,pk):
  c = pow(m,pk[1],pk[0])
  return c

def dec(c,sk):
  m = pow(c,sk[1],sk[0])
  return m

if __name__=='__main__':
  n_bit_length = 200
  (p,q,n) = generate_keys(n_bit_length)
  print("p = ",p)
  print("q= ", q)
  print("n= ",n)
  phi = (p-1)*(q-1)
  pk=(n,65537)
  sk=(n,pow(65537,-1,phi))

  m = 12
  c = enc(m,pk)
  print("ChiperText: ",c)

  m = dec(c,sk)
  print("Plaintext: ", m)

import rsa

(pk,sk) = rsa.newkeys(512)
print("pk= ",pk)
print("sk= ",sk)

m0 = "Test".encode('utf8')
c = rsa.encrypt(m0,pk)
m1 = rsa.decrypt(c,sk)

print(m1)
