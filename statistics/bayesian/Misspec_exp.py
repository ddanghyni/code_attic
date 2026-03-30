import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom, poisson
from scipy.special import gammaln
plt.rcParams["font.family"] = "AppleGothic" 

np.random.seed(42)

# =============================================
# 설정 — 극단적으로 overdispersion 크게!
# NB(r=2, p=0.3) : 평균 = 4.67, 분산 = 15.6
# 포아송은 평균=분산 이라 분산을 못 따라감
# =============================================
r_true = 2
p_true = 0.3
print(f"진짜 NB(r={r_true}, p={p_true})")
print(f"이론 평균: {r_true*(1-p_true)/p_true:.2f}")
print(f"이론 분산: {r_true*(1-p_true)/p_true**2:.2f}  ← 포아송이랑 엄청 다름!")

# =============================================
# n 크기별로 실험
# =============================================
n_list = [10, 100, 1000, 10000]
M = 5000
burn_in = 1000

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Misspecification: True=NB(r=2,p=0.3)  /  Wrong assumption=Poisson\n"
             "n increases → Poisson NEVER converges to true distribution", fontsize=13)

x_range = np.arange(0, 35)
true_pmf = nbinom.pmf(x_range, r_true, p_true)

for col, n in enumerate(n_list):

    # 데이터 생성
    x_data = nbinom.rvs(r_true, p_true, size=n)

    # =============================================
    # Poisson 가정 MCMC
    # =============================================
    def log_posterior_poisson(lam):
        if lam <= 0:
            return -np.inf
        log_lik = np.sum(x_data * np.log(lam) - lam)
        log_prior = (2-1)*np.log(lam) - 0.5*lam
        return log_lik + log_prior

    lam_curr = x_data.mean()
    lam_samples = []
    for _ in range(M):
        lam_prop = lam_curr + np.random.normal(0, 0.3)
        log_ratio = log_posterior_poisson(lam_prop) - log_posterior_poisson(lam_curr)
        if np.log(np.random.uniform()) < log_ratio:
            lam_curr = lam_prop
        lam_samples.append(lam_curr)
    lam_samples = np.array(lam_samples[burn_in:])

    # Posterior Predictive
    x_pred = np.array([poisson.rvs(lam) for lam in lam_samples])

    # =============================================
    # 윗줄: 사후분포
    # =============================================
    ax_top = axes[0, col]
    ax_top.hist(lam_samples, bins=40, density=True, alpha=0.7,
                color='tomato', label=f'Posterior λ\nmean={lam_samples.mean():.2f}')
    ax_top.axvline(r_true*(1-p_true)/p_true, color='black', linestyle='--',
                   linewidth=2, label=f'NB mean={r_true*(1-p_true)/p_true:.2f}')
    ax_top.set_title(f'n={n}\nPosterior of λ', fontsize=11)
    ax_top.set_xlabel('λ')
    ax_top.legend(fontsize=8)

    # =============================================
    # 아랫줄: 예측분포 vs 진짜
    # =============================================
    ax_bot = axes[1, col]

    # 진짜 분포
    ax_bot.bar(x_range - 0.2, true_pmf, width=0.35, alpha=0.8,
               color='green', label='True NB(r=2,p=0.3)')

    # 예측분포
    bins = np.arange(-0.5, 35.5)
    ax_bot.hist(x_pred, bins=bins, density=True, alpha=0.6,
                color='tomato', label='Predictive (Poisson 가정)')

    # KL divergence 계산
    pred_pmf, _ = np.histogram(x_pred, bins=bins, density=True)
    pred_pmf = pred_pmf + 1e-10
    true_pmf_clip = true_pmf + 1e-10
    kl = np.sum(true_pmf_clip * np.log(true_pmf_clip / pred_pmf))

    ax_bot.set_title(f'n={n}\nKL divergence={kl:.3f}', fontsize=11)
    ax_bot.set_xlabel('x')
    ax_bot.legend(fontsize=8)

plt.tight_layout()
plt.show()