#fast exp
import math

def FastExp(a,b,N):
  if b<0 : return False

  btemp = b
  x=a
  t=1

  while(btemp>=1):
    if(btemp%2):  #만약 btemp가 홀수일때 r곱해줌
      t=(t*x) %N
      btemp = btemp - 1


    x= (x**2) %N
    btemp = btemp >> 1

  return t

print(FastExp(5,10,1000))

