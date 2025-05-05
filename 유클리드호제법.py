#유클리드 호제법
import math

def GCD(a,b) :
  if a==b :
    return a

  if (a > b):
    if not (a % b) :
      return b
    else:
      return GCD(b,a % b)

  if(a < b) :
    if not (b % a) :
        return a
    else:
      return GCD(a, b % a)


a= 0xa4a700ac4f18634ac845739e0342cd8335bf6e0606ca9dc9d668abf9ed812e6d
b= 0xda7866632109e77f0d3c5bdd49031e6d9a940fcb7d29ea2f858b991d1f17cef8

e =0xc94d8b8223755fd121ec6aa0519ecee2

g=  hex(math.gcd(a,b))
print(g)

