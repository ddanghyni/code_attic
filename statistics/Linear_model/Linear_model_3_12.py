import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# =======================================================
# 1. 데이터 생성 
# 최적 uniform 0, 100 --> 이게 영향이 있나..? 기존에는 0, 10
# =======================================================
def generate_data(n=100, true_mu=0.0, true_a=2.0, true_b=3.0):
 
    S = np.random.uniform(0, 100, size=(n, 2))       
    D = cdist(S, S, metric='euclidean')             

    # 공분산: Cov(Y_i, Y_j) = a * exp(-||S_i - S_j|| / b)
    Sigma = true_a * np.exp(-D / true_b) 

    # Y ~ N_n(mu * 1, Sigma),  mu = 0 이므로 평균벡터 = 0벡터
    mu_vec = true_mu * np.ones(n)
    Y = np.random.multivariate_normal(mu_vec, Sigma)

    return S, D, Y   

# =======================================================
# 2. Log-likelihood
# 왜 수학을 잘해야하는지 알겠노
# =======================================================
def log_likelihood(theta, Y, D):
    a, b = theta
    mu = 0.0
    n = len(Y)

    # a, b는 양수여야 함
    if a <= 1e-10 or b <= 1e-10:
        return -np.inf

    Sigma = a * np.exp(-D / b) + 1e-8 * np.eye(n)

    try:
        # Cholesky ㅇdecomps
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        return -np.inf

    # mu = 0 이므로 residual = Y - 0 = Y
    residual = Y - mu * np.ones(n)   # 사실상 그냥 Y

    # L * alpha = residual  풀기  ->  alpha = L^{-1} residual
    alpha = np.linalg.solve(L, residual)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    quad_form = np.dot(alpha, alpha)

    return -0.5 * n * np.log(2 * np.pi) - 0.5 * log_det - 0.5 * quad_form


# =======================================================
# 3. Gradient compute
# 늇랩도 시간나면 ㄱㄱ 헤시안까지 구하면 더 잘 수렴할수도 있지 않을까..
# =======================================================
def compute_gradient(theta, Y, D, eps=1e-5):

    grad = np.zeros(2)   
    for k in range(2):   
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[k] += eps
        theta_minus[k] -= eps
        grad[k] = (log_likelihood(theta_plus, Y, D) -
                    log_likelihood(theta_minus, Y, D)) / (2 * eps)
    return grad


# =======================================================
# 4. UPdate parameter by Gradient Ascent
# =======================================================
def parameter_updating(theta_init, Y, D, LR, max_iter):
    theta = theta_init.copy()
    tol = 1e-6

    history_a = [theta[0]]
    history_b = [theta[1]]
    history_ll = [log_likelihood(theta, Y, D)]

    print("Gradient Ascent start")
    print(f"initial:  a = {theta[0]}, b = {theta[1]}")
    print(f"initial log-likelihood : {history_ll[0]:.4f}")
    print()

    prev_ll = history_ll[0]

    for t in range(max_iter):
        grad = compute_gradient(theta, Y, D)
        
        theta = theta + LR * grad
        
        current_ll = log_likelihood(theta, Y, D)

        history_a.append(theta[0])
        history_b.append(theta[1])
        history_ll.append(current_ll)

        if (t + 1) % 200 == 0:
            print(f"Iter {t+1:4d}: a={theta[0]:.4f}, "
                  f"b={theta[1]:.4f}, loglik={current_ll:.4f}")

        if abs(current_ll - prev_ll) < tol:
            print(f"\n수렴햇노 (iter {t+1})")
            break

        prev_ll = current_ll

    # ★ 반환값 3개: theta, hist_a, hist_b, hist_ll
    return theta, history_a, history_b, history_ll


# =======================================================
# 5. 실행
# 최적 n = 500, LR = 0.001 --> LR이 영향을 많이 줫나..? 기존에는 0.0001 --> 이게 좀 더 정교하게 학습을 시키니깐 좋지 않을까 했는디요..
# =======================================================
np.random.seed(202682114)

S, D, Y = generate_data(n=1000, true_mu=0.0, true_a=2.0, true_b=3.0)

theta_init = np.array([1.0, 2.0])  # [a, b] 초기값

theta_hat, hist_a, hist_b, hist_ll = parameter_updating(
    theta_init, Y, D, LR=0.001, max_iter=5000 
)

print(f"\nResults:")
print(f"  a  = {theta_hat[0]:.4f}  (참값: 2.0)")
print(f"  b  = {theta_hat[1]:.4f}  (참값: 3.0)")


# =======================================================
# 6. 시각화
# =======================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# (1) a 수렴
ax = axes[0]
ax.plot(hist_a, color='darkorange', lw=1.5)
ax.axhline(2.0, color='red', ls='--', lw=1, label='true = 2.0')
ax.set_xlabel('Iter')
ax.set_ylabel('a')
ax.set_title('converg')
ax.legend()
ax.grid(True, alpha=0.3)


# (2) b 수렴
ax = axes[1]
ax.plot(hist_b, color='seagreen', lw=1.5)
ax.axhline(3.0, color='red', ls='--', lw=1, label='true = 3.0')
ax.set_xlabel('Ite')
ax.set_ylabel('b')
ax.set_title('b converg')
ax.legend()
ax.grid(True, alpha=0.3)


# (3) log-likelihood 수렴
ax = axes[2]
ax.plot(hist_ll, color='purple', lw=1.5)
ax.set_xlabel('Iter')
ax.set_ylabel('Log-Likelihood')
ax.set_title('Log-Likelihood converg')
ax.grid(True, alpha=0.3)
plt.show()
