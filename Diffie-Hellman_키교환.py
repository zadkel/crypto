import math
import random

if __name__=='__main__':

  n=1019
  g=2
  q=1018

  x=random.randrange(1,n)
  X=pow(g,x,n)

  y=random.randrange(1,n)
  Y=pow(g,y,n)

  key_X = pow(Y,x,n)
  key_Y = pow(X,y,n)

  print(x,y, X, Y,key_X,key_Y)
