import numpy as np

class Polynomial:
    """
    다항식을 나타내는 클래스. Z_q[X] / (X^N + 1) 위에서의 연산을 지원해야 함.
    """
    def __init__(self, coefficients: list[int], N: int, q: int):
        self.coeffs = np.array(coefficients, dtype=object)
        self.N = N
        self.q = q

    def __add__(self, other: 'Polynomial') -> 'Polynomial':
        if not isinstance(other, Polynomial):
            raise TypeError("Polynomial can only be added with another Polynomial.")
        coeffs = (self.coeffs + other.coeffs) % self.q
        return Polynomial(coeffs.tolist(), self.N, self.q)

    def __mul__(self, other: 'Polynomial') -> 'Polynomial':
        if not isinstance(other, Polynomial):
            raise TypeError("Polynomial can only be multiplied with another Polynomial.")
        print("!! WARNING: Using placeholder for Polynomial multiplication. This is NOT cryptographically correct. !!")
        coeffs = (self.coeffs * other.coeffs) % self.q # 실제와 매우 다른, 개념적 표현일 뿐입니다.
        return Polynomial(coeffs.tolist(), self.N, self.q)

    def __repr__(self) -> str:
        return f"Polynomial({self.coeffs.tolist()}, N={self.N}, q={self.q})"

# --- CKKS 데이터 구조 클래스 ---
class CKKSParameters:
    def __init__(self, N: int, q_init: int, delta: float, sigma: float, num_levels: int = 1):
        self.N = N  # 다항식 차수 (X^N + 1)
        self.q_init = q_init  # 초기 (최상위 레벨) 암호문 계수 모듈러스
        self.delta = delta  # 스케일링 팩터
        self.sigma = sigma # 에러 분포의 표준 편차
        self.num_levels = num_levels # 지원하는 레벨 수 (실제 모듈러스 체인 관리는 생략)
        # 실제 시스템에서는 q_values = [q_L, ..., q_0] 형태의 모듈러스 리스트 관리 필요

class CKKSSecretKey:
    def __init__(self, s: Polynomial):
        self.s = s # 비밀키 다항식 s

class CKKSPublicKey:
    def __init__(self, pk0: Polynomial, pk1: Polynomial):
        self.pk0 = pk0 # 공개키의 첫 번째 요소 b = -a*s + e
        self.pk1 = pk1 # 공개키의 두 번째 요소 a

class CKKSCiphertext:
    def __init__(self, c0: Polynomial, c1: Polynomial, level: int, delta_at_encryption: float, q_at_encryption: int):
        self.c0 = c0
        self.c1 = c1
        self.level = level # 암호문의 현재 레벨
        self.delta_at_encryption = delta_at_encryption # 암호화 시점의 스케일 (리스케일링 추적에 필요)
        self.q_at_encryption = q_at_encryption # 암호화 시점의 모듈러스

# --- CKKS 엔진 클래스 ---
class CKKSEngine:
    def __init__(self, params: CKKSParameters, verbose: bool = False):
        self.params = params
        self.verbose = verbose
        if self.verbose:
            print(f">> Conceptual CKKS Engine initialized with N={params.N}, q_init={params.q_init}, delta={params.delta:.2f}\n")

    # --- "Private" Helper Methods ---
    def _sample_gaussian_polynomial(self, q_current: int) -> Polynomial:
        coeffs = np.round(np.random.normal(0, self.params.sigma, self.params.N)).astype(int) % q_current
        return Polynomial(coeffs.tolist(), self.params.N, q_current)

    def _sample_uniform_polynomial(self, q_current: int) -> Polynomial:
        coeffs = np.random.randint(0, q_current, self.params.N, dtype=object)
        return Polynomial(coeffs.tolist(), self.params.N, q_current)

    def _sample_ternary_polynomial(self, q_current: int) -> Polynomial:
        coeffs = np.random.randint(-1, 2, self.params.N)
        # Ternary 분포는 보통 작은 값을 가지므로, q가 매우 작지 않다면 %q가 큰 의미 없을 수 있음
        # 그러나 형식적으로는 계수가 q 미만이 되도록 해야 함.
        coeffs = coeffs % q_current
        return Polynomial(coeffs.tolist(), self.params.N, q_current)

    def _encode(self, message_vector: list[complex], delta_current: float, q_current: int) -> Polynomial:
        if self.verbose:
            print(f"Encoding {len(message_vector)} elements with delta={delta_current:.2f}")
        # 실제 CKKS 인코딩은 IFFT와 유사한 변환, 스케일링, 라운딩 등 복잡한 과정을 포함
        if len(message_vector) > self.params.N // 2:
            raise ValueError(f"Message vector length {len(message_vector)} cannot exceed N/2 = {self.params.N//2}")
        
        coeffs = [int(round(m.real * delta_current)) for m in message_vector]
        # 모듈러스 q_current에 대한 처리가 필요할 수 있으나, 인코딩 단계에서는 보통 q보다 작은 값을 다룸
        # 패딩 추가
        padded_coeffs = coeffs + [0] * (self.params.N - len(coeffs))
        return Polynomial(padded_coeffs, self.params.N, q_current)

    def _decode(self, poly: Polynomial, delta_current: float, target_length: int) -> list[complex]:
        if self.verbose:
            print(f"Decoding polynomial with delta={delta_current:.2f} to {target_length} elements")
        # 실제 CKKS 디코딩은 FFT와 유사한 변환, 스케일 축소 등을 포함
        num_coeffs_to_decode = min(target_length, self.params.N // 2, len(poly.coeffs))
        coeffs_to_decode = poly.coeffs[:num_coeffs_to_decode]
        return [complex(float(c) / delta_current) for c in coeffs_to_decode]

    # --- Public API Methods ---
    def create_secret_key(self) -> CKKSSecretKey:
        """비밀키를 생성"""
        # 실제 라이브러리는 특정 레벨이나 전체 모듈러스 체인에 대한 키를 생성할 수 있음.
        # 여기서는 초기 모듈러스 q_init을 사용.
        s = self._sample_ternary_polynomial(self.params.q_init) # 또는 가우시안
        if self.verbose:
            print("Secret key created.")
        return CKKSSecretKey(s)

    def create_public_key(self, sk: CKKSSecretKey) -> CKKSPublicKey:
        """공개키를 생성."""
        s_poly = sk.s
        q_current = self.params.q_init # 공개키는 보통 최상위 레벨 모듈러스로 생성

        a = self._sample_uniform_polynomial(q_current)
        e = self._sample_gaussian_polynomial(q_current)

        # pk0 = -(a*s_poly) + e (mod q_current)
        neg_a_coeffs = (-np.array(a.coeffs)) % q_current
        neg_a = Polynomial(neg_a_coeffs.tolist(), self.params.N, q_current)
        
        # pk0_poly = (neg_a * s_poly) + e
        dummy_neg_as_coeffs = (np.array(neg_a.coeffs) * np.array(s_poly.coeffs)) % q_current
        dummy_pk0_coeffs = (dummy_neg_as_coeffs + np.array(e.coeffs)) % q_current
        pk0_poly = Polynomial(dummy_pk0_coeffs.tolist(), self.params.N, q_current)

        if self.verbose:
            print("Public key created.")
        return CKKSPublicKey(pk0_poly, a)

    def encrypt(self, message_vector: list[complex], pk: CKKSPublicKey, level: int = 0) -> CKKSCiphertext:
        """메시지 벡터 암호화"""
        # 실제 레벨 시스템에서는 level에 따라 q_current, delta_current가 달라짐
        q_current = self.params.q_init # 또는 get_q_for_level(level)
        delta_current = self.params.delta # 또는 get_delta_for_level(level)

        if self.verbose:
            print(f"Encrypting for level {level} with q={q_current}, delta={delta_current:.2f}")

        m_poly = self._encode(message_vector, delta_current, q_current)

        u = self._sample_ternary_polynomial(q_current)
        e1 = self._sample_gaussian_polynomial(q_current)
        e2 = self._sample_gaussian_polynomial(q_current)

        # c0 = pk.pk0 * u + e1 + m_poly (mod q_current)
        # c1 = pk.pk1 * u + e2 (mod q_current)
        
        # pk0_u = pk.pk0 * u
        # c0 = pk0_u + e1 + m_poly
        # pk1_u = pk.pk1 * u
        # c1 = pk1_u + e2

        dummy_pk0_u_coeffs = (np.array(pk.pk0.coeffs) * np.array(u.coeffs)) % q_current
        dummy_c0_coeffs = (dummy_pk0_u_coeffs + np.array(e1.coeffs) + np.array(m_poly.coeffs)) % q_current
        c0 = Polynomial(dummy_c0_coeffs.tolist(), self.params.N, q_current)

        dummy_pk1_u_coeffs = (np.array(pk.pk1.coeffs) * np.array(u.coeffs)) % q_current
        dummy_c1_coeffs = (dummy_pk1_u_coeffs + np.array(e2.coeffs)) % q_current
        c1 = Polynomial(dummy_c1_coeffs.tolist(), self.params.N, q_current)

        return CKKSCiphertext(c0, c1, level, delta_current, q_current)

    def decrypt(self, ciphertext: CKKSCiphertext, sk: CKKSSecretKey, target_length: int = None) -> list[complex]:
        """(단일 사용자용) 암호문을 복호화합니다."""
        q_current = ciphertext.q_at_encryption # 암호화 시 사용된 모듈러스 사용
        delta_current = ciphertext.delta_at_encryption

        if target_length is None:
            target_length = self.params.N // 2 # 기본값

        if self.verbose:
            print(f"Decrypting ciphertext at level {ciphertext.level} with q={q_current}, delta to use={delta_current:.2f}")

        # noisy_m_poly = ciphertext.c0 + (ciphertext.c1 * sk.s) (mod q_current)

        # c1_s = ciphertext.c1 * sk.s
        # noisy_m_poly = ciphertext.c0 + c1_s
        
        dummy_c1_s_coeffs = (np.array(ciphertext.c1.coeffs) * np.array(sk.s.coeffs)) % q_current
        dummy_noisy_m_coeffs = (np.array(ciphertext.c0.coeffs) + dummy_c1_s_coeffs) % q_current
        noisy_m_poly = Polynomial(dummy_noisy_m_coeffs.tolist(), self.params.N, q_current)
        
        decrypted_vector = self._decode(noisy_m_poly, delta_current, target_length)
        return decrypted_vector

    def example(self, amin: int = -10, amax: int = 10, num_elements: int = None) -> list[complex]:
        """예시 평문 벡터를 생성"""
        if num_elements is None:
            num_elements = self.params.N // 2 
        # 간단히 실수부만 있는 복소수 생성
        return [complex(np.random.uniform(amin, amax)) for _ in range(num_elements)]

    def absmax_error(self, m_original: list[complex], m_decrypted: list[complex]) -> float:
        """원본과 복호화된 메시지 간의 최대 절대 오차를 계산"""
        if len(m_original) == 0 and len(m_decrypted) == 0:
            return 0.0
        if len(m_original) != len(m_decrypted):
            # 길이가 다를 경우, 더 짧은 쪽을 0으로 패딩하여 비교
            # 실제 애플리케이션에서는 상황에 맞게 처리해야 함
            max_len = max(len(m_original), len(m_decrypted))
            m_o_padded = m_original + [0j] * (max_len - len(m_original))
            m_d_padded = m_decrypted + [0j] * (max_len - len(m_decrypted))
            errors = [abs(orig - dec) for orig, dec in zip(m_o_padded, m_d_padded)]
        else:
            errors = [abs(orig - dec) for orig, dec in zip(m_original, m_decrypted)]
        
        return max(errors) if errors else 0.0


# --- 예시 ---
if __name__ == '__main__':
    print("====="*20)
    print("Conceptual CKKS Engine Demo (Single Party)")
    print("WARNING: This code is for educational purposes ONLY and NOT secure or fully correct.")
    print("Polynomial operations are placeholders and not cryptographically sound.")
    print("====="*20 + "\n")

    # 파라미터 설정

    N_val = 2**10 # 실제 라이브러리는 2**12 ~ 2**15 이상 사용
    q_initial = 1099511627776 # 예시: 2^40 근처의 값
    delta_val = 2**10 # 예시 스케일 팩터
    sigma_val = 3.2 # 표준 에러 편차

    params = CKKSParameters(N=N_val, q_init=q_initial, delta=delta_val, sigma=sigma_val, num_levels=1) # 1개 레벨만 가정
    
    # 엔진 생성
    engine = CKKSEngine(params, verbose=True)

    # 3. 키 생성
    sk = engine.create_secret_key()
    pk = engine.create_public_key(sk)
    print("\n-----"*20)

    # 암호화할 메시지 생성
    num_elements_to_encrypt = 10 # 암호화할 메시지 벡터 길이
    original_message = engine.example(amin=-5, amax=5, num_elements=num_elements_to_encrypt)
    print(f"Original message (first 5 elements): {original_message[:5]}")
    print("-----"*20)

    # 암호화 (레벨 0에서 암호화한다고 가정)
    ciphertext = engine.encrypt(original_message, pk, level=0)
    print("Encryption complete.")
    # print(f"Ciphertext c0 (first 5 coeffs): {ciphertext.c0.coeffs[:5]}") 
    print("-----"*20)

    # 복호화
    decrypted_message = engine.decrypt(ciphertext, sk, target_length=num_elements_to_encrypt)
    print(f"Decrypted message (first 5 elements): {decrypted_message[:5]}")
    print("-----"*20)

    # 오차 확인
    error = engine.absmax_error(original_message, decrypted_message)
    print(f"Max absolute error: {error:.15e}")
    print("====="*20)
