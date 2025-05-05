import math

def EEA(a,b):
  r0 = max(a,b)
  r1 = min(a,b)
  s0=1
  s1=0
  t0=0
  t1=1

  while(r1):
    q = r0 // r1
    if(q> r0/2):
      q=q-r1

    s=s0-q*s1
    t=t0-q*t1
    r=r0-q*r1

    s0=s1
    t0=t1
    r0=r1
    s1=s
    t1=t
    r1=r

  return r0,s0,t0


print(EEA(294,91))
