import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, gamma, gaussian_kde
plt.rcParams["font.family"] = "AppleGothic" 

# =============================================
# 설정
# =============================================
lambda_true = 4.0       # 진짜 lambda (현실에선 모름)
n = 100                  # 데이터 개수
alpha, beta = 2.0, 1.0  # prior Gamma(alpha, beta) 파라미터
M = 10000               # MCMC 샘플 수
np.random.seed(42)

# =============================================
# ① 데이터 생성 (진짜 lambda에서)
# =============================================
x_data = np.random.poisson(lambda_true, size=n)
print(f"진짜 lambda: {lambda_true}")
print(f"데이터 평균: {x_data.mean():.3f}")

# =============================================
# ② 사후분포 수식 (Gamma-Poisson 켤레)
# 사실 이 경우엔 사후분포가 Gamma로 딱 떨어지지만
# MCMC로 해보는 실험임!
# =============================================
alpha_post = alpha + x_data.sum()
beta_post = beta + n
print(f"\n사후분포 Gamma({alpha_post:.1f}, {beta_post:.1f})")
print(f"사후분포 평균: {alpha_post/beta_post:.3f}")

# =============================================
# ③ MCMC (Metropolis-Hastings)
# =============================================
def log_posterior(lam):
    if lam <= 0:
        return -np.inf
    log_likelihood = np.sum(x_data * np.log(lam) - lam)
    log_prior = (alpha - 1) * np.log(lam) - beta * lam
    return log_likelihood + log_prior

samples = []
lam_current = 1.0

for _ in range(M):
    lam_proposed = lam_current + np.random.normal(0, 0.5)
    log_ratio = log_posterior(lam_proposed) - log_posterior(lam_current)
    if np.log(np.random.uniform()) < log_ratio:
        lam_current = lam_proposed
    samples.append(lam_current)

# burn-in 제거
lambda_samples = np.array(samples[3000:])
print(f"\nMCMC 사후분포 평균: {lambda_samples.mean():.3f}")

# =============================================
# ④ Posterior Predictive 샘플 생성
# 모든 lambda* 에 대해 Poisson에 꽂기
# =============================================
x_tilde = np.array([np.random.poisson(lam) for lam in lambda_samples])

# =============================================
# ⑤ 시각화
# =============================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"Bayesian Poisson (λ_true={lambda_true}, n={n})", fontsize=14)

# --- plot 1: 사후분포 ---
ax1 = axes[0]
ax1.hist(lambda_samples, bins=50, density=True, alpha=0.5, color='steelblue', label='MCMC samples')

# 켤레로 구한 진짜 사후분포
lam_range = np.linspace(lambda_samples.min(), lambda_samples.max(), 300)
ax1.plot(lam_range, gamma.pdf(lam_range, a=alpha_post, scale=1/beta_post),
         'r-', lw=2, label=f'True Posterior\nGamma({alpha_post:.0f},{beta_post:.0f})')
ax1.axvline(lambda_true, color='black', linestyle='--', label=f'λ_true={lambda_true}')
ax1.set_title('① 사후분포 p(λ|x)')
ax1.set_xlabel('λ')
ax1.legend()

# --- plot 2: 데이터 분포 vs 예측분포 ---
ax2 = axes[1]
x_range = np.arange(0, 15)

# 진짜 데이터 분포 (lambda_true 고정)
true_pmf = poisson.pmf(x_range, lambda_true)
ax2.bar(x_range - 0.2, true_pmf, width=0.4, alpha=0.6,
        color='green', label=f'데이터 분포\nPoisson(λ_true={lambda_true})')

# 예측분포 (histogram)
ax2.hist(x_tilde, bins=np.arange(-0.5, 15.5), density=True, alpha=0.6,
         color='orange', label='Posterior Predictive')
ax2.set_title('② 데이터 분포 vs 예측분포')
ax2.set_xlabel('x')
ax2.legend()

# --- plot 3: 데이터 수에 따라 예측분포가 수렴하는지 ---
ax3 = axes[2]
for n_sub in [5, 30, 200]:
    x_sub = x_data[:n_sub] if n_sub <= n else np.random.poisson(lambda_true, n_sub)
    a_p = alpha + x_sub.sum()
    b_p = beta + len(x_sub)
    lam_sub = np.random.gamma(a_p, 1/b_p, size=5000)
    x_pred = np.array([np.random.poisson(l) for l in lam_sub])
    ax3.hist(x_pred, bins=np.arange(-0.5, 15.5), density=True, alpha=0.4, label=f'n={n_sub}')

ax3.bar(x_range, true_pmf, width=0.3, alpha=0.8,
        color='black', label=f'데이터 분포\n(λ_true={lambda_true})')
ax3.set_title('③ 데이터 수 증가 → 예측분포 수렴')
ax3.set_xlabel('x')
ax3.legend()

plt.tight_layout()
plt.show()