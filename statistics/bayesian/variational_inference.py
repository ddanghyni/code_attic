"""
비켤레 (non-conjugate) Bayesian: t-분포 likelihood
- 모델:  μ ~ N(0, τ₀²),  xᵢ | μ ~ t_ν(μ, σ²)  ← 헤비 테일!
- 정답 사후분포: 닫힌형 없음 (켤레 깨짐)
- ELBO: 닫힌형 없음 → Monte Carlo + Reparameterization trick
- MCMC를 ground truth로 사용해서 VI 정확도 평가
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.manual_seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────────────────────
# 1. 모델 설정
# ──────────────────────────────────────────────────────────────
TRUE_MU = 2.5  # 진짜 평균 (시뮬용)
sigma2 = 1.0  # likelihood 스케일
nu = 3.0  # t-분포 자유도 (작을수록 더 헤비테일)
tau0_2 = 10.0  # prior 분산
mu_0 = 0.0  # prior 평균
N = 50  # 데이터 수

# t-분포에서 데이터 샘플링 (특이값 포함 가능 — 헤비테일!)
t_dist = torch.distributions.StudentT(df=nu, loc=TRUE_MU, scale=math.sqrt(sigma2))
x_data = t_dist.sample((N,))
print(
    f"데이터 통계: mean={x_data.mean():.3f}, std={x_data.std():.3f}, "
    f"min={x_data.min():.3f}, max={x_data.max():.3f}"
)
print(f"(헤비테일이라 표본평균이 진짜 평균 {TRUE_MU}에서 좀 벗어날 수 있음)")


# ──────────────────────────────────────────────────────────────
# 2. 로그 확률 함수 (모두 PyTorch 자동미분 가능)
# ──────────────────────────────────────────────────────────────
def log_prior(mu):
    """log p(μ) = log N(μ; μ_0, τ₀²)"""
    return -0.5 * (mu - mu_0) ** 2 / tau0_2


def log_likelihood(x, mu):
    """log p(xᵢ | μ) = log t_ν(xᵢ; μ, σ²)  — 비켤레의 원인!"""
    return -((nu + 1) / 2) * torch.log(1 + (x - mu) ** 2 / (nu * sigma2))


def log_joint(mu):
    """log p(x, μ) = log p(μ) + Σ log p(xᵢ | μ)"""
    return log_prior(mu) + log_likelihood(x_data, mu).sum()


# ──────────────────────────────────────────────────────────────
# 3. VI: ELBO를 Monte Carlo + Reparameterization으로 추정
# ──────────────────────────────────────────────────────────────
m = torch.tensor(0.0, requires_grad=True)
log_s2 = torch.tensor(0.0, requires_grad=True)
N_SAMPLES = 20  # Monte Carlo 샘플 수


def elbo_mc(m, log_s2, n_samples=N_SAMPLES):
    """
    ELBO ≈ (1/L) Σ [log p(x, μ⁽ˡ⁾) - log q(μ⁽ˡ⁾)]
    여기서 μ⁽ˡ⁾ = m + s·ε⁽ˡ⁾,  ε⁽ˡ⁾ ~ N(0,1)  (reparam trick)
    """
    s = torch.exp(0.5 * log_s2)
    eps = torch.randn(n_samples)  # ε ~ N(0,1)
    mu_samples = m + s * eps  # reparam: μ = m + s·ε

    # log p(x, μ) 항 (각 샘플마다)
    log_p = torch.stack([log_joint(mu_l) for mu_l in mu_samples])

    # log q(μ): N(m, s²) 의 밀도
    log_q = -0.5 * (mu_samples - m) ** 2 / s**2 - 0.5 * log_s2

    return (log_p - log_q).mean()


# ──────────────────────────────────────────────────────────────
# 4. 최적화 + ELBO 히스토리 기록
# ──────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam([m, log_s2], lr=0.05)
elbo_history = []
m_history = []
s2_history = []

print("\nVI 학습 시작:")
print(f"{'step':>6} {'m':>10} {'s²':>10} {'ELBO':>12}")
for step in range(3000):
    optimizer.zero_grad()
    loss = -elbo_mc(m, log_s2)
    loss.backward()
    optimizer.step()

    elbo_history.append(-loss.item())
    m_history.append(m.item())
    s2_history.append(torch.exp(log_s2).item())

    if step % 300 == 0:
        print(
            f"{step:>6d} {m.item():>10.4f} {torch.exp(log_s2).item():>10.4f} {-loss.item():>12.4f}"
        )

vi_m = m.item()
vi_s2 = torch.exp(log_s2).item()
print(f"\n[VI 결과] q(μ) = N({vi_m:.4f}, {vi_s2:.4f})")

# ──────────────────────────────────────────────────────────────
# 5. Ground truth: MCMC로 진짜 사후분포 샘플링 (Metropolis-Hastings)
#    닫힌형 정답이 없으니 MCMC를 정답으로 사용
# ──────────────────────────────────────────────────────────────
print("\nMCMC ground truth 생성 중 (Metropolis-Hastings)...")


def log_joint_np(mu):
    """numpy 버전 — MCMC용"""
    mu_t = torch.tensor(mu)
    return log_joint(mu_t).item()


# Random-walk Metropolis-Hastings
mcmc_samples = []
mu_curr = 0.0
n_mcmc = 50000
n_burnin = 5000
n_accept = 0
for i in range(n_mcmc + n_burnin):
    mu_prop = mu_curr + np.random.randn() * 0.2  # 제안 분포
    log_ratio = log_joint_np(mu_prop) - log_joint_np(mu_curr)
    if np.log(np.random.rand()) < log_ratio:
        mu_curr = mu_prop
        n_accept += 1
    if i >= n_burnin:
        mcmc_samples.append(mu_curr)

mcmc_samples = np.array(mcmc_samples)
mcmc_mean = mcmc_samples.mean()
mcmc_var = mcmc_samples.var()
print(f"MCMC 수락률: {n_accept / (n_mcmc + n_burnin):.3f}")
print(f"[MCMC ground truth] mean={mcmc_mean:.4f}, var={mcmc_var:.4f}")

print(f"\n=== 비교 ===")
print(f"VI:   q(μ) = N({vi_m:.4f}, {vi_s2:.4f})")
print(f"MCMC: μ ~  N-like(mean={mcmc_mean:.4f}, var={mcmc_var:.4f})")
print(f"차이:   Δμ={abs(vi_m - mcmc_mean):.4f}, Δσ²={abs(vi_s2 - mcmc_var):.4f}")

# ──────────────────────────────────────────────────────────────
# 6. 시각화
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# (a) ELBO 상승 곡선
axes[0, 0].plot(
    elbo_history, color="steelblue", alpha=0.5, label="per-step ELBO (with MC noise)"
)
axes[0, 0].set_xlabel("step")
axes[0, 0].set_ylabel("ELBO")
axes[0, 0].set_title("ELBO over training (monotonically increasing)")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# (b) 변분 파라미터 m의 학습 곡선
axes[0, 1].plot(m_history, color="darkgreen")
axes[0, 1].axhline(
    mcmc_mean, color="red", linestyle="--", label=f"MCMC mean = {mcmc_mean:.3f}"
)
axes[0, 1].axhline(TRUE_MU, color="gray", linestyle=":", label=f"true mu = {TRUE_MU}")
axes[0, 1].set_xlabel("step")
axes[0, 1].set_ylabel("m (variational mean)")
axes[0, 1].set_title("Variational mean convergence")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# (c) 변분 파라미터 s²의 학습 곡선
axes[1, 0].plot(s2_history, color="purple")
axes[1, 0].axhline(
    mcmc_var, color="red", linestyle="--", label=f"MCMC var = {mcmc_var:.4f}"
)
axes[1, 0].set_xlabel("step")
axes[1, 0].set_ylabel("s² (variational variance)")
axes[1, 0].set_title("Variational variance convergence")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# (d) 사후분포 비교: MCMC 히스토그램 vs VI Gaussian
x_range = np.linspace(
    mcmc_mean - 4 * math.sqrt(mcmc_var), mcmc_mean + 4 * math.sqrt(mcmc_var), 500
)
axes[1, 1].hist(
    mcmc_samples,
    bins=80,
    density=True,
    alpha=0.5,
    color="red",
    label="MCMC samples (ground truth)",
)
vi_pdf = np.exp(-0.5 * (x_range - vi_m) ** 2 / vi_s2) / np.sqrt(2 * np.pi * vi_s2)
axes[1, 1].plot(x_range, vi_pdf, "b-", linewidth=2.5, label="VI approximation N(m, s²)")
axes[1, 1].axvline(TRUE_MU, color="gray", linestyle=":", label=f"true mu = {TRUE_MU}")
axes[1, 1].set_xlabel("mu")
axes[1, 1].set_ylabel("density")
axes[1, 1].set_title("Posterior: MCMC (truth) vs VI (Gaussian approx)")
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    "/Users/sanghyun/Desktop/code_attic/code_attic/statistics/bayesian/VI_png/vi_nonconjugate.png",
    dpi=100,
)
print("\n결과 그림: vi_nonconjugate.png")
