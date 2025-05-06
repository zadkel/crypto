import numpy as np

N_DIM = 10      # 비밀키 s의 차원, 행렬 A의 열 개수
M_SAMPLES = 40  # LWE 샘플의 개수, 행렬 A의 행 개수 (보통 M > N * log(Q) 필요)
Q_MODULUS = 127 # 모듈러스 q (계산의 기준이 되는 소수 또는 정수)
                  # 실제 시스템에서는 2^10 이상, 또는 더 큰 소수 사용

ERROR_BOUND = 1   # 오류 벡터 e의 각 요소는 {-1, 0, 1} 중 하나가 됨
                  # 이 값이 너무 크면 복호화 실패 확률이 높아짐

def sample_small_vector(size: int, bound: int, q: int) -> np.ndarray:
    """작은 정수들로 이루어진 벡터를 샘플링 (비밀키 s, 오류 벡터 e 용)"""
    return np.random.randint(-bound, bound + 1, size) % q

def sample_binary_vector(size: int) -> np.ndarray:
    """이진 벡터 (0 또는 1)를 샘플링 (암호화 시 r 용)"""
    return np.random.randint(0, 2, size)

def lwe_keygen() -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    LWE 공개키와 비밀키를 생성
    Returns:
        tuple: (공개키, 비밀키)
               공개키: (행렬 A, 벡터 p)의 튜플
               비밀키: 벡터 s
    """
    # 비밀키 s (작은 정수 벡터)
    s_sk = sample_small_vector(N_DIM, ERROR_BOUND, Q_MODULUS)

    # 공개 행렬 A (Z_q 위의 랜덤 행렬)
    A_pk = np.random.randint(0, Q_MODULUS, size=(M_SAMPLES, N_DIM))

    # 오류 벡터 e (작은 정수 벡터)
    e_error = sample_small_vector(M_SAMPLES, ERROR_BOUND, Q_MODULUS)

    p_pk = (A_pk @ s_sk + e_error) % Q_MODULUS
    
    public_key = (A_pk, p_pk)
    secret_key = s_sk
    
    return public_key, secret_key

def lwe_encrypt(public_key: tuple[np.ndarray, np.ndarray], message_bit: int) -> tuple[np.ndarray, int]:
    """
    LWE 공개키를 사용하여 단일 비트 메시지를 암호화
    Args:
        public_key: (행렬 A, 벡터 p).
        message_bit: 암호화할 비트 (0 또는 1).
    Returns:
        tuple: 암호문 (벡터 u, 스칼라 정수 v).
    """
    if message_bit not in [0, 1]:
        raise ValueError("메시지는 0 또는 1이어야 합니다.")
        
    A_pk, p_pk = public_key
    
    # 1. 암호화를 위한 무작위 이진 벡터 r 생성 (r in {0,1}^M_SAMPLES)
    r_encryption_randomness = sample_binary_vector(M_SAMPLES)
        
    # 2. 암호문 u 계산: u = (A^T * r) mod q
    # A_pk.T는 A_pk의 전치 행렬
    u_ciphertext = (A_pk.T @ r_encryption_randomness) % Q_MODULUS
    
    # 3. 메시지 비트 인코딩: 0 -> 0, 1 -> floor(q/2)
    encoded_message = message_bit * (Q_MODULUS // 2)
    
    # 4. 암호문 v 계산: v = (p^T * r + encoded_message) mod q
    # p_pk.T @ r_encryption_randomness는 스칼라 값 (내적)
    # (작은 오류를 이 단계에 추가하는 변형도 있음)
    v_ciphertext_scalar_part = p_pk.T @ r_encryption_randomness
    v_ciphertext = (v_ciphertext_scalar_part + encoded_message) % Q_MODULUS
    
    ciphertext = (u_ciphertext, v_ciphertext)
    # print(f"메시지 {message_bit} 암호화 완료:\n  u (암호문 일부): {ciphertext[0][:3]}...\n  v (암호문): {ciphertext[1]}")
    return ciphertext

def lwe_decrypt(secret_key: np.ndarray, ciphertext: tuple[np.ndarray, int]) -> int:
    """
    LWE 비밀키를 사용하여 암호문을 복호화
    Args:
        secret_key: 비밀키 벡터 s
        ciphertext: 암호문 (벡터 u, 스칼라 정수 v)
    Returns:
        int: 복호화된 비트 (0 또는 1), 또는 복호화 실패 시 -1
    """
    s_sk = secret_key
    u_ciphertext, v_ciphertext = ciphertext
    
    # v - s^T * u 계산 (mod q)
    # s_sk.T @ u_ciphertext 는 스칼라 값 (내적)
    # 이 결과는 (e^T * r + message_bit * floor(q/2)) mod q 와 유사해야 함
    decryption_intermediate_value = (v_ciphertext - (s_sk.T @ u_ciphertext)) % Q_MODULUS
    
    # 결과값이 0에 가까운지, 아니면 floor(q/2)에 가까운지 판별
    # 노이즈 e^T * r 이 q/4 보다 작아야 정확한 복호화 가능
    q_half_floored = Q_MODULUS // 2
    
    # 복호화된 값이 q_half_floored 를 기준으로 어느 쪽에 더 가까운지 확인
    # dist_to_zero: 0으로부터의 거리 (모듈러 연산 고려)
    # dist_to_q_half: q_half_floored로부터의 거리 (모듈러 연산 고려)
    
    # decryption_intermediate_value는 이미 [0, Q_MODULUS-1] 범위에 있음
    if decryption_intermediate_value > q_half_floored:
        # 값이 q/2보다 크면, 0으로부터의 거리는 q - 값
        dist_to_zero = Q_MODULUS - decryption_intermediate_value
    else:
        dist_to_zero = decryption_intermediate_value
        
    dist_to_q_half = abs(decryption_intermediate_value - q_half_floored)
    
    if dist_to_zero < dist_to_q_half:
        # 0에 더 가까우면 메시지는 0
        return 0
    else:
        # floor(q/2)에 더 가까우면 메시지는 1
        return 1


if __name__ == "__main__":
    print(f"파라미터: N={N_DIM}, M={M_SAMPLES}, Q={Q_MODULUS}, ErrorBound={ERROR_BOUND}\n")

    # 1. 키 생성
    pk, sk = lwe_keygen()
    print("키 생성 완료.")
    # print(f"  공개키 A (일부):\n{pk[0][:2,:3]}...") # 너무 길어서 주석 처리
    # print(f"  비밀키 s (일부): {sk[:3]}...")

    # 2. 암호화할 메시지 비트
    message_to_encrypt_0 = 0
    message_to_encrypt_1 = 1

    # 3. 메시지 0 암호화 및 복호화
    print(f"\n--- 메시지 {message_to_encrypt_0} 암호화 및 복호화 ---")
    ciphertext_0 = lwe_encrypt(pk, message_to_encrypt_0)
    print(f"  암호문 (u 일부, v): ({ciphertext_0[0][:3]}..., {ciphertext_0[1]})")
    decrypted_message_0 = lwe_decrypt(sk, ciphertext_0)
    print(f"  복호화된 메시지: {decrypted_message_0}")
    if decrypted_message_0 == message_to_encrypt_0:
        print(" 결과: 성공!")
    else:
        print(" 결과: 실패!")

    # 4. 메시지 1 암호화 및 복호화
    print(f"\n--- 메시지 {message_to_encrypt_1} 암호화 및 복호화 ---")
    ciphertext_1 = lwe_encrypt(pk, message_to_encrypt_1)
    print(f"  암호문 (u 일부, v): ({ciphertext_1[0][:3]}..., {ciphertext_1[1]})")
    decrypted_message_1 = lwe_decrypt(sk, ciphertext_1)
    print(f"  복호화된 메시지: {decrypted_message_1}")
    if decrypted_message_1 == message_to_encrypt_1:
        print(" 결과: 성공!")
    else:
        print(" 결과: 실패!")
        
    print("M * ERROR_BOUND 가 Q / 4 보다 충분히 작아야 안정적인 복호화가 가능합니다.")
    print(f"현재 M * ERROR_BOUND = {M_SAMPLES * ERROR_BOUND}, Q / 4 = {Q_MODULUS / 4:.2f}")
    if M_SAMPLES * ERROR_BOUND >= Q_MODULUS / 4:
        print("파라미터 조건이 복호화 실패를 유발할 수 있습니다.")
