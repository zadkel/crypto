#여러개의 값의 modulat inverse를 한 번에 구하는 알고리즘

import math

def FastModInv(arr,n):
      num = len(arr)
      if num == 0:
        return []
      
      b=[1]*num
      b[0] = arr[0]
      for i in range(1, num):
        b[i] = (b[i-1] * arr[i]) % n
      ainv = [0] * num

      try:
          inv_last_b = pow(b[num-1], -1, n)
      except ValueError:
          # 역원이 존재하지 않는 경우 처리 (n과 b[num-1]이 서로소가 아님)
          print(f"Error: Modular inverse does not exist for {b[num-1]} mod {n}")
          return None # 또는 다른 오류 처리

      # 역방향으로 개별 역원 계산
      # 마지막 원소부터 계산
      for i in range(num - 1, 0, -1):
          # ainv[i] = (b[i]의 역원) * b[i-1] mod n
          ainv[i] = (inv_last_b * b[i-1]) % n
          # 다음 계산을 위해 b[i-1]의 역원을 계산
          # inv(b[i-1]) = inv(b[i]) * arr[i] mod n
          inv_last_b = (inv_last_b * arr[i]) % n

      # 첫 번째 원소의 역원 계산
      ainv[0] = inv_last_b % n

      return ainv


arr=(2,3,5,7)
n=11

print(FastModInv(arr,n))
