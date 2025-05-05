#Goal : 정렬 알고리즘은 내장함수, 바이너리 서치는 구현 과제해야함
import time
import numpy as np

#binary search
def BINARY_SEARCH(arr, first, last, target):
  if (first > last):
    return -1

  mid = (first+last)//2

  if(arr[mid] == target):
    return mid
  elif(arr[mid]>target):
    return BINARY_SEARCH(arr, first, mid-1, target)
  else:
    return BINARY_SEARCH(arr, mid+1, last, target)

#Find g^x = h (mod q)

def BSGS(g,h,q):
  n = q+1
  t = int(math.sqrt(q))
  gt = pow(g,t,n)
  print("t=",t)

  GS = [[0,1],[1,gt]]

  #Giant Step List
  for k in range(2,t+1):
    GS.append( [k, (GS[k-1][1]*gt)%n] )
  print("Giant Step : ", GS)

  #Sorting Giant Step List
  GS.sort(key = lambda  x : x[1])
  GS_arr = np.array(GS)[:,1]


  #Baby Step
  hi = h

  print("baby step : ", end="")
  for i in range(t):
    if i !=0:
      hi = (hi*g)%n
      print("[", i,",", hi,"]", end="")
      #Binary Search
      index = BINARY_SEARCH(GS_arr,0,len(GS_arr)-1,hi)

      if(index != -1):
        k=GS[index][0]
        x=(k * t - i)%q
        print("")
        print( "x = " , k , "*" , t, "-" , i , "(mod ",q,")=" , x)
        return x
  return None

if __name__ == "__main__":
  start_time=time.time()
  x = BSGS(g=2, h=17, q=28)
  print('x=log_g(h)=',x)
  print("run time : ", time.time() - start_time)
