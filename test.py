import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# =======================================================
# 1. 데이터 생성 (mu = 0 고정)
# =======================================================
def generate_data(n=100, true_mu=0.0, true_a=2.0, true_b=3.0):
    S = np.random.uniform(0, 10, size=(n, 2))
    D = cdist(S, S, metric='euclidean')
    Sigma = true_a * np.exp(-D / true_b) + 1e-8 * np.eye(n)
    mu_vec = true_mu * np.ones(n)
    Y = np.random.multivariate_normal(mu_vec, Sigma)
    return S, D, Y


# =======================================================
# 2. Log-likelihood (mu = 0 고정)
# =======================================================
def log_likelihood(theta, Y, D):
    a, b = theta
    n = len(Y)
    if a <= 1e-10 or b <= 1e-10:
        return -np.inf
    Sigma = a * np.exp(-D / b) + 1e-8 * np.eye(n)
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        return -np.inf
    alpha = np.linalg.solve(L, Y)  # mu=0이므로 residual = Y
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    quad_form = np.dot(alpha, alpha)
    return -0.5 * n * np.log(2 * np.pi) - 0.5 * log_det - 0.5 * quad_form


# =======================================================
# 3. Gradient (수치 미분)
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
# 4. Gradient Ascent + Backtracking Line Search
#
#    왜 line search가 필요한가?
#    - 고정 LR은 gradient 크기에 따라 너무 크거나 작을 수 있음
#    - a와 b의 gradient 스케일이 다르면 한쪽은 잘 되는데
#      한쪽은 안 움직이거나 발산함
#    - line search: "이 방향으로 갈건데, 얼마나 갈지는
#      실제로 log-likelihood가 올라가는지 보고 결정"
# =======================================================
def parameter_updating(theta_init, Y, D, lr_init=0.01, max_iter=5000):
    theta = theta_init.copy()
    tol = 1e-8
    lr = lr_init

    history_a = [theta[0]]
    history_b = [theta[1]]
    history_ll = [log_likelihood(theta, Y, D)]

    print("Gradient Ascent (with line search) start")
    print(f"initial:  a = {theta[0]:.4f}, b = {theta[1]:.4f}")
    print(f"initial log-likelihood : {history_ll[0]:.4f}\n")

    prev_ll = history_ll[0]

    for t in range(max_iter):
        grad = compute_gradient(theta, Y, D)

        # --- Backtracking Line Search ---
        # 기본 아이디어: lr로 한 스텝 갔는데 log-likelihood가
        # 오히려 줄어들면? lr을 반으로 줄여서 다시 시도.
        # 올라갈 때까지 반복.
        new_theta = theta + lr * grad
        new_ll = log_likelihood(new_theta, Y, D)

        while new_ll < prev_ll and lr > 1e-12:
            lr *= 0.5  # 스텝 사이즈 줄이기
            new_theta = theta + lr * grad
            new_ll = log_likelihood(new_theta, Y, D)

        theta = new_theta
        lr = min(lr * 1.2, 0.1)  # lr 조금씩 복구 (너무 작아지면 안 되니까)

        history_a.append(theta[0])
        history_b.append(theta[1])
        history_ll.append(new_ll)

        if (t + 1) % 50 == 0:
            print(f"Iter {t+1:4d}: a={theta[0]:.4f}, "
                  f"b={theta[1]:.4f}, lr={lr:.6f}, loglik={new_ll:.4f}")

        if abs(new_ll - prev_ll) < tol:
            print(f"\nConverged! (iter {t+1})")
            break

        prev_ll = new_ll

    return theta, history_a, history_b, history_ll


# =======================================================
# 5. 실행
# =======================================================
np.random.seed(42)

S, D, Y = generate_data(n=500, true_mu=0.0, true_a=2.0, true_b=3.0)

theta_init = np.array([1.0, 2.0])  # [a, b] 초기값

theta_hat, hist_a, hist_b, hist_ll = parameter_updating(
    theta_init, Y, D, lr_init=0.01, max_iter=5000
)

print(f"\nResults:")
print(f"  a  = {theta_hat[0]:.4f}  (참값: 2.0)")
print(f"  b  = {theta_hat[1]:.4f}  (참값: 3.0)")


# =======================================================
# 6. 시각화
# =======================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
ax.plot(hist_a, color='darkorange', lw=1.5)
ax.axhline(2.0, color='red', ls='--', lw=1, label='true = 2.0')
ax.set_xlabel('Iteration')
ax.set_ylabel('a')
ax.set_title('a convergence')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(hist_b, color='seagreen', lw=1.5)
ax.axhline(3.0, color='red', ls='--', lw=1, label='true = 3.0')
ax.set_xlabel('Iteration')
ax.set_ylabel('b')
ax.set_title('b convergence')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(hist_ll, color='purple', lw=1.5)
ax.set_xlabel('Iteration')
ax.set_ylabel('Log-Likelihood')
ax.set_title('Log-Likelihood convergence')
ax.grid(True, alpha=0.3)
