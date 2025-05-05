#The Rho algorithm of DL

u=(37,71,76)
v=(34,69,18)
gj=[]

def walk(x,a,b,q,n,u,v,gj):
  s=x%4

  if (s==0):
    x = (x*x)%n
    a = (2*a)%q
    b = (2*b)%q
  else:
    x=(x*gj[s-1])%n
    a=(a+u[s-1])%q
    b=(b+v[s-1])%q
  return x,a,b


#Goal : g^x=h인 x찾기, q는 order
def PollardRho(g,h,q,n):
  res = 0

  for s in range(3):
    k=(pow(g,u[s],n) *pow(h,v[s],n))%n
    gj.append(k)

  print(u,v,gj)

  (x1,a1,b1) = (g,1,0)
  (x2,a2,b2) = walk(x1,a1,b1,q,n,u,v,gj)

  print("i   xi  a1  b1  x2  a2  b2")
  print("------------------------------")
  print('{0:03d}  {1:03d}  {2:03d}  {3:03d}  {4:03d}  {5:03d}'.format(x1,a1,b1,x2,a2,b2))
#  print(x1,a1,b1,x2,a2,b2)

  while(x1!=x2):
    (x1,a1,b1) = walk(x1,a1,b1,q,n,u,v,gj)
    (x2,a2,b2) = walk(x2,a2,b2,q,n,u,v,gj)
    (x2,a2,b2) = walk(x2,a2,b2,q,n,u,v,gj)
    print('{0:03d}  {1:03d}  {2:03d}  {3:03d}  {4:03d}  {5:03d}'.format(x1,a1,b1,x2,a2,b2))


  if ((b1-b2)%q!=0):
    res = ((a2-a1)*pow(b1-b2,-1,q))%q
  return res

if __name__ == '__main__':
  n=809
  q=101
  g=89
  h=799
  res = PollardRho(g,h,q,n)
  print(res)
