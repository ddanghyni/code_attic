# %% [markdown]
# # 부산 지역 카페/음식점 평점 분석
# ## Part 1: 데이터 로드, 전처리, 탐색적 자료분석(EDA)
# 
# **대주제:** 부산 지역 카페/음식점의 평점은 가격, 위치, 리뷰 수, 업종에 따라  
# 어떻게 달라지며, 이러한 요인들의 설명력은 어디까지인가?
# 
# **데이터 출처:** Google Places API (New)  
# **수집 URL:** https://developers.google.com/maps/documentation/places/web-service  
# **수집 날짜:** 2026-03-17  

# %% [markdown]
# ---
# ## 0. 환경 설정

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── 한글 폰트 설정 (환경에 맞게 수정) ──
# Windows: "Malgun Gothic"
# Mac: "AppleGothic"  
# Linux: "Noto Sans CJK KR"
#plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["font.family"] = "AppleGothic" 
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

# 시각화 스타일
sns.set_style("whitegrid")
sns.set_palette("Set2")

print("환경 설정 완료")

# %% [markdown]
# ---
# ## 1. 데이터 로드
# 
# Google Places API (New)를 통해 부산 16개 행정구의 음식점과 카페 데이터를 수집하였다.  
# API에서 제공하는 필드 중 평점(rating), 리뷰 수(review_count), 가격대(price_level),  
# 업종(business_type), 주소(address) 등을 활용한다.

# %%
# 원본 데이터 로드
df_raw = pd.read_csv("/Users/sanghyun/Desktop/code_attic/output/google_places_20260317_090717.csv") 

print(f"원본 데이터: {df_raw.shape[0]}행 × {df_raw.shape[1]}열")
print(f"\n컬럼 목록:")
for i, col in enumerate(df_raw.columns, 1):
    print(f"  {i:2d}. {col:<20s} (dtype: {df_raw[col].dtype})")

# %%
# 상위 5행 확인
df_raw.head()

# %% [markdown]
# ---
# ## 2. 전처리
# 
# ### 2.1 분석 대상 선정 기준
# 
# 본 연구에서는 **가격대(price_level)** 정보가 존재하는 가게만을 분석 대상으로 선정하였다.  
# 그 이유는 다음과 같다:
# 
# 1. 가격은 본 연구의 핵심 독립변수 중 하나이므로, 가격 정보가 없으면 소주제 1(가격대별 평점)과  
#    소주제 5(통합 모델)에서 해당 관측치를 활용할 수 없다.
# 2. 가격대 정보가 있는 가게는 Google에서 충분한 정보가 축적된 가게이므로,  
#    평점과 리뷰 수의 신뢰도가 상대적으로 높다.
# 3. 분석 전체에서 동일한 데이터셋을 사용함으로써 소주제 간 일관성을 확보한다.
# 
# 또한, 부산광역시의 공식 16개 행정구·군에 해당하지 않는 관측치는 제외하였다.

# %%
# ── 2.1 분석 대상 필터링 ──

# 부산 16개 행정구·군
valid_districts = [
    "해운대구", "부산진구", "금정구", "남구", "수영구",
    "중구", "동래구", "사하구", "북구", "사상구",
    "연제구", "영도구", "강서구", "동구", "서구", "기장군"
]

# 조건: (1) price_level 존재 (2) rating 존재 (3) 유효 행정구
df = df_raw[
    df_raw["price_level"].notna() &
    df_raw["rating"].notna() &
    df_raw["district"].isin(valid_districts)
].copy()

print(f"필터링 결과:")
print(f"  원본: {len(df_raw)}개")
print(f"  → price_level 존재 + rating 존재 + 유효 행정구: {len(df)}개")
print(f"  제거된 행: {len(df_raw) - len(df)}개")

# %% [markdown]
# ### 2.2 변수 확인 및 정리

# %%
# ── 2.2 결측치 확인 ──
print("필터링 후 결측치 현황:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "  → 주요 분석 변수에 결측치 없음")

print(f"\n각 변수별 유효 데이터 수:")
print(f"  rating:       {df['rating'].notna().sum()}")
print(f"  review_count: {df['review_count'].notna().sum()}")
print(f"  price_level:  {df['price_level'].notna().sum()}")
print(f"  district:     {df['district'].notna().sum()}")
print(f"  business_type:{df['business_type'].notna().sum()}")

# %% [markdown]
# ### 2.3 변수 변환이 필요한 이유 확인
# 
# 변수를 변환하기 **전에**, 현재 데이터가 어떤 상태인지 먼저 시각적으로 확인한다.

# %%
# ── 2.3-1 price_level 원본 상태 확인 ──
# "왜 범주화가 필요한가?"를 데이터로 보여주기

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (좌) price_level 원본 분포 — 문제점이 보인다
price_raw_counts = df["price_level"].value_counts().sort_index()
labels_raw = {1.0: "1\n(INEXPENSIVE)", 2.0: "2\n(MODERATE)", 3.0: "3\n(EXPENSIVE)", 4.0: "4\n(VERY_EXPENSIVE)"}
colors_raw = ["#70AD47", "#5B9BD5", "#ED7D31", "#C00000"]
bars = axes[0].bar([labels_raw[k] for k in price_raw_counts.index], price_raw_counts.values, 
                    color=colors_raw, edgecolor="white")
for bar, val in zip(bars, price_raw_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 10, f"{val}개\n({val/len(df)*100:.1f}%)", 
                 ha="center", fontsize=10, fontweight="bold")
axes[0].set_ylabel("가게 수", fontsize=11)
axes[0].set_title("(a) price_level 원본 분포\n→ 4(VERY_EXPENSIVE)가 5개뿐", fontsize=12)

# (우) 범주화 후 — 고가(3) + 최고가(4)를 합침
# 먼저 범주화 수행
def categorize_price(level):
    if level == 1:
        return "저가"
    elif level == 2:
        return "중가"
    else:  # 3, 4
        return "고가"

df["price_category"] = df["price_level"].apply(categorize_price)
price_cat_counts = df["price_category"].value_counts().reindex(["저가", "중가", "고가"])
colors_cat = ["#70AD47", "#5B9BD5", "#ED7D31"]
bars = axes[1].bar(price_cat_counts.index, price_cat_counts.values, color=colors_cat, edgecolor="white")
for bar, val in zip(bars, price_cat_counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 10, f"{val}개\n({val/len(df)*100:.1f}%)", 
                 ha="center", fontsize=10, fontweight="bold")
axes[1].set_ylabel("가게 수", fontsize=11)
axes[1].set_title("(b) 범주화 후: 고가 + 최고가 합침\n→ 3그룹으로 비교 가능", fontsize=12)

plt.suptitle("price_level 변환: 원본 → 범주화", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# %%
# ── 2.3-1b Class Imbalance 수치 비교 ──
# 합치기 전후로 클래스 불균형이 얼마나 개선되었는지 정량적으로 확인

n = len(df)

# 합치기 전 (4 클래스)
before_counts = df["price_level"].value_counts().sort_index()
props_before = before_counts / n
ir_before = before_counts.max() / before_counts.min()  # Imbalance Ratio
entropy_before = -np.sum(props_before * np.log(props_before))
balance_before = entropy_before / np.log(len(before_counts))  # 1이면 완전 균형

# 합친 후 (3 클래스)
after_counts = df["price_category"].value_counts().reindex(["저가", "중가", "고가"])
props_after = after_counts / n
ir_after = after_counts.max() / after_counts.min()
entropy_after = -np.sum(props_after * np.log(props_after))
balance_after = entropy_after / np.log(len(after_counts))

# 비교 테이블
print("Class Imbalance 비교 (합치기 전 vs 후)")
print(f"{'─'*65}")
print(f"{'지표':<30} {'합치기 전(4cls)':<18} {'합친 후(3cls)':<15}")
print(f"{'─'*65}")
print(f"{'클래스 수':<30} {len(before_counts):<18} {len(after_counts):<15}")
print(f"{'최소 클래스 크기':<30} {before_counts.min():<18} {after_counts.min():<15}")
print(f"{'최대/최소 비율 (IR)':<30} {ir_before:<18.1f} {ir_after:<15.1f}")
print(f"{'Balance Ratio (0~1)':<30} {balance_before:<18.4f} {balance_after:<15.4f}")
print(f"{'─'*65}")
print(f"  IR: {ir_before:.0f}:1 → {ir_after:.0f}:1 (↓ 개선)")
print(f"  Balance Ratio: {balance_before:.3f} → {balance_after:.3f} (↑ 개선)")

# 시각화: 비교 막대그래프
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

metrics = ["Imbalance Ratio\n(최대/최소, 낮을수록 균형)", "Balance Ratio\n(0~1, 높을수록 균형)"]
before_vals = [ir_before, balance_before]
after_vals = [ir_after, balance_after]

x = np.arange(len(metrics))
w = 0.3
bars1 = axes[0].bar(x - w/2, before_vals, w, label="합치기 전 (4cls)", color="#C44E52", alpha=0.8)
bars2 = axes[0].bar(x + w/2, after_vals, w, label="합친 후 (3cls)", color="#5B9BD5", alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics, fontsize=10)
axes[0].set_ylabel("값", fontsize=11)
axes[0].set_title("(a) 불균형 지표 비교", fontsize=12)
axes[0].legend()
# 값 표시
for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 f"{bar.get_height():.1f}", ha="center", fontsize=9)
for bar in bars2:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 f"{bar.get_height():.1f}", ha="center", fontsize=9)

# 비율 파이차트 (합친 후)
axes[1].pie(after_counts.values, labels=[f"{k}\n({v}개, {v/n*100:.1f}%)" for k, v in after_counts.items()],
            colors=["#70AD47", "#5B9BD5", "#ED7D31"], autopct="", startangle=90, 
            wedgeprops=dict(edgecolor="white", linewidth=2))
axes[1].set_title("(b) 합친 후 비율 분포\n→ 중가 74.8% 여전히 지배적", fontsize=12)

plt.suptitle("Class Imbalance 분석", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"\n※ 합친 후에도 중가(74.8%)가 지배적인 class imbalance가 존재함")
print(f"  → 소주제 1 분석 시 비모수 검정(Kruskal-Wallis) 사용 및 해석에 주의 필요")

print(f"\n결정: VERY_EXPENSIVE(4)가 5개뿐이므로 EXPENSIVE(3)와 합쳐 '고가'로 범주화")
print(f"  저가 (1): {(df['price_category']=='저가').sum()}개")
print(f"  중가 (2): {(df['price_category']=='중가').sum()}개")
print(f"  고가 (3+4): {(df['price_category']=='고가').sum()}개")

# %%
# ── 2.3-2 review_count 원본 상태 확인 + 로그 변환 ──
# "왜 로그 변환이 필요한가?"를 데이터로 보여주기

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# (좌) 원본 분포 — 극단적 right-skewed
axes[0].hist(df["review_count"], bins=50, color="#5B9BD5", edgecolor="white", alpha=0.85)
axes[0].axvline(df["review_count"].mean(), color="red", linestyle="--", 
                label=f'평균: {df["review_count"].mean():.0f}')
axes[0].axvline(df["review_count"].median(), color="orange", linestyle="--", 
                label=f'중앙값: {df["review_count"].median():.0f}')
axes[0].set_xlabel("리뷰 수", fontsize=11)
axes[0].set_ylabel("빈도", fontsize=11)
axes[0].set_title(f"(a) 원본 분포\n왜도(skewness) = {df['review_count'].skew():.2f}", fontsize=12)
axes[0].legend(fontsize=9)

# (중) 로그 변환 후 — 훨씬 대칭적
df["log_review_count"] = np.log1p(df["review_count"])
axes[1].hist(df["log_review_count"], bins=30, color="#70AD47", edgecolor="white", alpha=0.85)
axes[1].axvline(df["log_review_count"].mean(), color="red", linestyle="--", 
                label=f'평균: {df["log_review_count"].mean():.2f}')
axes[1].axvline(df["log_review_count"].median(), color="orange", linestyle="--", 
                label=f'중앙값: {df["log_review_count"].median():.2f}')
axes[1].set_xlabel("log(1 + 리뷰 수)", fontsize=11)
axes[1].set_ylabel("빈도", fontsize=11)
axes[1].set_title(f"(b) 로그 변환 후\n왜도(skewness) = {df['log_review_count'].skew():.2f}", fontsize=12)
axes[1].legend(fontsize=9)

# (우) Q-Q plot으로 정규성 비교
from scipy.stats import probplot
probplot(df["log_review_count"], dist="norm", plot=axes[2])
axes[2].set_title("(c) 로그 변환 후 Q-Q Plot\n→ 정규분포에 근접", fontsize=12)

plt.suptitle("review_count 변환: 원본 → log(1+x)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"변환 전: 왜도 = {df['review_count'].skew():.2f} (심한 right-skew)")
print(f"변환 후: 왜도 = {df['log_review_count'].skew():.2f} (대칭에 가까움)")
print(f"→ 회귀분석 등에서 log_review_count를 사용하면 모델 가정에 더 적합")

# %%
# ── 2.3-3 리뷰 수 구간 변수 생성 ──
# 소주제 2에서 구간별 비교에 사용

df["review_group"] = pd.cut(
    df["review_count"],
    bins=[0, 50, 100, 300, 1000, np.inf],
    labels=["~50", "51~100", "101~300", "301~1000", "1000+"],
    include_lowest=True,
)

fig, ax = plt.subplots(figsize=(8, 4))
rg_counts = df["review_group"].value_counts().sort_index()
bars = ax.bar(rg_counts.index, rg_counts.values, color="#5B9BD5", edgecolor="white")
for bar, val in zip(bars, rg_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 5, str(val), ha="center", fontweight="bold")
ax.set_xlabel("리뷰 수 구간", fontsize=11)
ax.set_ylabel("가게 수", fontsize=11)
ax.set_title("리뷰 수 구간별 가게 수", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

print("리뷰 수 구간별 분포:")
print(rg_counts.to_string())

# %% [markdown]
# ### 2.4 독립변수 간 다중공선성 사전 확인
# 
# 소주제 5에서 다중회귀 모델을 적합하기 전에,  
# 독립변수 간 강한 상관이 있는지 사전에 확인한다.  
# 강한 상관(|r| > 0.7)이 있으면 회귀 계수의 추정이 불안정해질 수 있다.

# %%
# 수치형 독립변수 간 상관계수
indep_vars = ["price_level", "review_count", "log_review_count"]
indep_corr = df[indep_vars].corr()

print("독립변수 간 Pearson 상관계수:")
print(indep_corr.round(3).to_string())

print(f"\n주요 쌍별 상관:")
print(f"  price_level ↔ review_count:     r = {df['price_level'].corr(df['review_count']):.3f}")
print(f"  price_level ↔ log_review_count: r = {df['price_level'].corr(df['log_review_count']):.3f}")

# review_count와 log_review_count는 변환 관계이므로 당연히 높음 → 둘 중 하나만 모델에 사용
print(f"  review_count ↔ log_review_count: r = {df['review_count'].corr(df['log_review_count']):.3f} (변환 관계, 둘 중 하나만 사용)")

print(f"\n판단:")
print(f"  price_level ↔ log_review_count: |r| = {abs(df['price_level'].corr(df['log_review_count'])):.3f} < 0.3")
print(f"  → 독립변수 간 심각한 다중공선성 없음")
print(f"  → 소주제 5에서 VIF로 정식 진단 예정")

# %% [markdown]
# ### 2.5 전처리 완료 요약

# %%
# ── 최종 데이터 요약 ──
print("=" * 55)
print("전처리 완료 데이터 요약")
print("=" * 55)
print(f"  총 관측치:     {len(df)}개")
print(f"  행정구:        {df['district'].nunique()}개")
print(f"  업종:          음식점 {(df['business_type']=='음식점').sum()}개 / 카페 {(df['business_type']=='카페').sum()}개")
print(f"")
print(f"  [종속변수]")
print(f"  rating:        {df['rating'].min()} ~ {df['rating'].max()} (평균: {df['rating'].mean():.3f}, 중앙값: {df['rating'].median()})")
print(f"")
print(f"  [독립변수]")
print(f"  price_category: 저가 {(df['price_category']=='저가').sum()} / 중가 {(df['price_category']=='중가').sum()} / 고가 {(df['price_category']=='고가').sum()}")
print(f"  review_count:   {df['review_count'].min()} ~ {df['review_count'].max()} (중앙값: {df['review_count'].median():.0f})")
print(f"  district:       {df['district'].nunique()}개 행정구 (최소 {df['district'].value_counts().min()}개 ~ 최대 {df['district'].value_counts().max()}개)")
print(f"  business_type:  음식점 / 카페")

# ── 분석에 사용할 변수 정리 ──
print(f"\n분석에 사용할 변수:")
print(f"  종속변수: rating (평점, 1.0~5.0)")
print(f"  독립변수: price_category, log_review_count, district, business_type")
print(f"  파생변수: price_category, log_review_count, review_group")

# %% [markdown]
# ---
# ## 3. 탐색적 자료분석 (EDA)
# 
# 본격적인 소주제 분석에 앞서, 데이터의 전반적인 구조와 분포를 탐색한다.

# %% [markdown]
# ### 3.1 종속변수: 평점(rating) 분포

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (좌) 히스토그램
axes[0].hist(df["rating"], bins=25, color="#5B9BD5", edgecolor="white", alpha=0.85)
axes[0].axvline(df["rating"].mean(), color="red", linestyle="--", linewidth=1.5, 
                label=f'평균: {df["rating"].mean():.2f}')
axes[0].axvline(df["rating"].median(), color="orange", linestyle="--", linewidth=1.5, 
                label=f'중앙값: {df["rating"].median():.1f}')
axes[0].set_xlabel("평점", fontsize=12)
axes[0].set_ylabel("빈도", fontsize=12)
axes[0].set_title("(a) 평점 분포", fontsize=13)
axes[0].legend(fontsize=10)

# (우) 박스플롯
bp = axes[1].boxplot(df["rating"], vert=True, patch_artist=True, 
                      boxprops=dict(facecolor="#5B9BD5", alpha=0.7))
axes[1].set_ylabel("평점", fontsize=12)
axes[1].set_title("(b) 평점 박스플롯", fontsize=13)
axes[1].set_xticklabels(["전체"])

# 통계량 텍스트
stats_text = (f'n = {len(df)}\n'
              f'평균 = {df["rating"].mean():.2f}\n'
              f'표준편차 = {df["rating"].std():.2f}\n'
              f'왜도 = {df["rating"].skew():.2f}')
axes[1].text(1.3, df["rating"].min() + 0.1, stats_text, fontsize=10, 
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle("Figure 1. 종속변수(평점) 분포", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# **해석:**  
# - 평점은 평균 4.14, 중앙값 4.2로 **좌편향(left-skewed)** 분포를 보인다.
# - 대부분의 가게가 3.5~4.5 사이에 밀집되어 있으며, 2점대 이하는 극소수이다.
# - 이는 온라인 평점의 일반적 특성(긍정 편향, 천장효과)과 일치한다.

# %% [markdown]
# ### 3.2 독립변수 분포

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# (a) 가격대 분포
price_counts = df["price_category"].value_counts().reindex(["저가", "중가", "고가"])
colors_price = ["#70AD47", "#5B9BD5", "#ED7D31"]
bars = axes[0, 0].bar(price_counts.index, price_counts.values, color=colors_price, edgecolor="white")
for bar, val in zip(bars, price_counts.values):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, val + 10, str(val), ha="center", fontweight="bold")
axes[0, 0].set_ylabel("가게 수", fontsize=11)
axes[0, 0].set_title("(a) 가격대별 가게 수", fontsize=12)

# (b) 리뷰 수 분포 (로그 스케일)
axes[0, 1].hist(df["log_review_count"], bins=30, color="#5B9BD5", edgecolor="white", alpha=0.85)
axes[0, 1].axvline(df["log_review_count"].median(), color="orange", linestyle="--", 
                    label=f'중앙값: {df["review_count"].median():.0f}건')
axes[0, 1].set_xlabel("log(1 + 리뷰 수)", fontsize=11)
axes[0, 1].set_ylabel("빈도", fontsize=11)
axes[0, 1].set_title("(b) 리뷰 수 분포 (로그 변환 후)", fontsize=12)
axes[0, 1].legend(fontsize=10)

# (c) 업종 분포
bt_counts = df["business_type"].value_counts()
colors_bt = ["#5B9BD5", "#ED7D31"]
bars = axes[1, 0].bar(bt_counts.index, bt_counts.values, color=colors_bt, edgecolor="white")
for bar, val in zip(bars, bt_counts.values):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, val + 10, f"{val}\n({val/len(df)*100:.1f}%)", 
                     ha="center", fontsize=10)
axes[1, 0].set_ylabel("가게 수", fontsize=11)
axes[1, 0].set_title("(c) 업종별 가게 수", fontsize=12)

# (d) 행정구별 가게 수
dist_counts = df["district"].value_counts().sort_values()
axes[1, 1].barh(dist_counts.index, dist_counts.values, color="#5B9BD5", edgecolor="white")
for i, val in enumerate(dist_counts.values):
    axes[1, 1].text(val + 2, i, str(val), va="center", fontsize=9)
axes[1, 1].set_xlabel("가게 수", fontsize=11)
axes[1, 1].set_title("(d) 행정구별 가게 수", fontsize=12)

plt.suptitle("Figure 2. 독립변수 분포", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# **해석:**  
# - **(a) 가격대:** 중가(MODERATE)가 1,021개(74.8%)로 대다수를 차지한다. 저가 285개(20.9%), 고가 59개(4.3%).
# - **(b) 리뷰 수:** 로그 변환 후 비교적 대칭적 분포를 보인다. 원본 중앙값 221건으로, 리뷰가 어느 정도 축적된 가게들이다.
# - **(c) 업종:** 음식점(62.3%)이 카페(37.7%)보다 많지만, 두 그룹 모두 충분한 표본 크기를 가진다.
# - **(d) 행정구:** 16개 구·군 모두 포함되어 있으며, 최소 51개(연제구)~최대 171개(부산진구)로 분석에 충분하다.

# %% [markdown]
# ### 3.3 변수 간 상관관계

# %%
# 수치형 변수 간 상관행렬
corr_vars = ["rating", "review_count", "log_review_count", "price_level"]
corr_labels = ["평점", "리뷰 수", "log 리뷰 수", "가격대"]
corr_matrix = df[corr_vars].corr()

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
            square=True, vmin=-1, vmax=1, ax=ax,
            xticklabels=corr_labels, yticklabels=corr_labels,
            linewidths=0.5)
ax.set_title("Figure 3. 수치형 변수 간 상관행렬 (Pearson)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# 주요 상관계수 출력
print("주요 상관관계:")
print(f"  평점 ↔ 리뷰 수:     r = {corr_matrix.loc['rating','review_count']:.3f}")
print(f"  평점 ↔ log 리뷰 수: r = {corr_matrix.loc['rating','log_review_count']:.3f}")
print(f"  평점 ↔ 가격대:      r = {corr_matrix.loc['rating','price_level']:.3f}")
print(f"  리뷰 수 ↔ 가격대:   r = {corr_matrix.loc['review_count','price_level']:.3f}")

# %% [markdown]
# **해석:**  
# - 평점과 다른 변수들 간의 상관관계는 전반적으로 **약하다** (|r| < 0.15).
# - 평점과 리뷰 수는 약한 음의 상관(r ≈ -0.10)으로, 리뷰가 많을수록 평점이 약간 낮아지는 경향이 있다.
# - 이는 리뷰가 많이 쌓일수록 다양한 의견이 반영되어 극단적 고평점이 줄어드는 효과일 수 있다.
# - 가격대와 평점 사이에는 거의 상관이 없다 (r ≈ 0.01).
# - **→ 개별 변수만으로 평점을 설명하기 어려울 수 있음을 시사한다.**

# %% [markdown]
# ### 3.4 주요 이변량 관계 탐색

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (a) 가격대별 평점
order_price = ["저가", "중가", "고가"]
sns.boxplot(data=df, x="price_category", y="rating", order=order_price,
            palette=["#70AD47", "#5B9BD5", "#ED7D31"], ax=axes[0], width=0.5)
# 평균 표시
for i, cat in enumerate(order_price):
    mean_val = df[df["price_category"] == cat]["rating"].mean()
    axes[0].scatter(i, mean_val, color="red", s=60, zorder=5, marker="D")
    axes[0].text(i + 0.15, mean_val, f"{mean_val:.2f}", color="red", fontsize=10, fontweight="bold")
axes[0].set_xlabel("가격대", fontsize=11)
axes[0].set_ylabel("평점", fontsize=11)
axes[0].set_title("(a) 가격대별 평점", fontsize=12)

# (b) 업종별 평점
sns.boxplot(data=df, x="business_type", y="rating",
            palette=["#5B9BD5", "#ED7D31"], ax=axes[1], width=0.4)
for i, bt in enumerate(df["business_type"].unique()):
    mean_val = df[df["business_type"] == bt]["rating"].mean()
    axes[1].scatter(i, mean_val, color="red", s=60, zorder=5, marker="D")
    axes[1].text(i + 0.12, mean_val, f"{mean_val:.2f}", color="red", fontsize=10, fontweight="bold")
axes[1].set_xlabel("업종", fontsize=11)
axes[1].set_ylabel("평점", fontsize=11)
axes[1].set_title("(b) 업종별 평점", fontsize=12)

# (c) 리뷰 수 vs 평점 산점도
axes[2].scatter(df["log_review_count"], df["rating"], alpha=0.25, s=12, color="#5B9BD5")
# 추세선
z = np.polyfit(df["log_review_count"], df["rating"], 1)
p_line = np.poly1d(z)
x_line = np.linspace(df["log_review_count"].min(), df["log_review_count"].max(), 100)
r_val = df["log_review_count"].corr(df["rating"])
axes[2].plot(x_line, p_line(x_line), color="red", linewidth=2, label=f"r = {r_val:.3f}")
axes[2].set_xlabel("log(1 + 리뷰 수)", fontsize=11)
axes[2].set_ylabel("평점", fontsize=11)
axes[2].set_title("(c) 리뷰 수 vs 평점", fontsize=12)
axes[2].legend(fontsize=10)

plt.suptitle("Figure 4. 주요 독립변수와 평점의 관계", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# **해석:**  
# - **(a)** 저가, 중가, 고가 간 평점의 중앙값과 평균이 매우 유사하다. 가격대에 따른 뚜렷한 차이는 보이지 않는다.
# - **(b)** 음식점과 카페의 평점 분포도 유사하나, 음식점이 약간 더 넓은 분산을 보인다.
# - **(c)** 리뷰 수와 평점 사이에 약한 음의 관계가 관찰된다. 리뷰가 적은 가게에서 평점의 변동 폭이 크다.

# %% [markdown]
# ### 3.5 행정구별 평점 분포

# %%
# 평균 평점 순으로 정렬
district_order = df.groupby("district")["rating"].mean().sort_values(ascending=False).index.tolist()

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# (a) 박스플롯
sns.boxplot(data=df, x="district", y="rating", order=district_order,
            palette="coolwarm_r", ax=axes[0], width=0.6)
axes[0].axhline(df["rating"].mean(), color="red", linestyle="--", alpha=0.5, 
                label=f'전체 평균: {df["rating"].mean():.2f}')
axes[0].set_xlabel("행정구", fontsize=11)
axes[0].set_ylabel("평점", fontsize=11)
axes[0].set_title("(a) 행정구별 평점 분포", fontsize=12)
axes[0].tick_params(axis="x", rotation=45)
axes[0].legend(fontsize=9)

# (b) 평균 + 95% 신뢰구간
dist_stats = df.groupby("district")["rating"].agg(["mean", "std", "count"])
dist_stats["se"] = dist_stats["std"] / np.sqrt(dist_stats["count"])
dist_stats["ci95"] = dist_stats["se"] * 1.96
dist_stats = dist_stats.sort_values("mean", ascending=True)

axes[1].barh(range(len(dist_stats)), dist_stats["mean"],
             xerr=dist_stats["ci95"], color="#5B9BD5", edgecolor="white",
             capsize=3, alpha=0.8)
axes[1].set_yticks(range(len(dist_stats)))
axes[1].set_yticklabels(dist_stats.index)
axes[1].axvline(df["rating"].mean(), color="red", linestyle="--", alpha=0.5,
                label=f'전체 평균: {df["rating"].mean():.2f}')
axes[1].set_xlabel("평균 평점 (95% 신뢰구간)", fontsize=11)
axes[1].set_title("(b) 행정구별 평균 평점과 95% 신뢰구간", fontsize=12)
axes[1].legend(fontsize=9)

plt.suptitle("Figure 5. 행정구별 평점 분포", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# 수치 요약
print("행정구별 요약 (평균 평점 순):")
summary = df.groupby("district")["rating"].agg(["count", "mean", "median", "std"]).round(3)
print(summary.sort_values("mean", ascending=False).to_string())

# %% [markdown]
# **해석:**  
# - 강서구(4.21), 금정구(4.18), 남구(4.18)이 상대적으로 높은 평균 평점을 보인다.
# - 동구(4.02), 사상구(4.06)는 상대적으로 낮은 평균 평점을 보인다.
# - 다만 대부분의 행정구가 4.0~4.2 범위 안에 있어 차이가 크지는 않다.
# - 95% 신뢰구간이 상당 부분 겹치는 것으로 보아, 통계적 유의성은 별도 검정이 필요하다.

# %% [markdown]
# ### 3.6 업종 × 가격대 교차 분석

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) 업종 × 가격대별 평점
sns.boxplot(data=df, x="price_category", y="rating", hue="business_type",
            order=["저가", "중가", "고가"], palette=["#5B9BD5", "#ED7D31"],
            ax=axes[0], width=0.6)
axes[0].set_xlabel("가격대", fontsize=11)
axes[0].set_ylabel("평점", fontsize=11)
axes[0].set_title("(a) 업종 × 가격대별 평점", fontsize=12)
axes[0].legend(title="업종", fontsize=10)

# (b) 업종 × 리뷰구간별 평점
review_order = ["~50", "51~100", "101~300", "301~1000", "1000+"]
sns.pointplot(data=df, x="review_group", y="rating", hue="business_type",
              order=review_order, palette=["#5B9BD5", "#ED7D31"],
              ax=axes[1], dodge=True, markers=["o", "s"], linestyles=["-", "--"],
              errorbar="ci", capsize=0.1)
axes[1].set_xlabel("리뷰 수 구간", fontsize=11)
axes[1].set_ylabel("평균 평점", fontsize=11)
axes[1].set_title("(b) 업종 × 리뷰 수 구간별 평균 평점", fontsize=12)
axes[1].legend(title="업종", fontsize=10)

plt.suptitle("Figure 6. 업종과 다른 변수의 교호작용 탐색", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# **해석:**  
# - **(a)** 저가와 중가에서는 음식점과 카페의 평점이 유사하나, 고가에서는 차이가 나타날 수 있다 (단, 고가 표본이 적어 주의 필요).
# - **(b)** 리뷰 수가 적은 구간(~50)에서 카페의 평점이 음식점보다 약간 높으며, 리뷰가 1,000건 이상인 구간에서는 두 업종 모두 평점이 하락하는 경향이 있다.
# - **→ 업종에 따라 평점 형성 패턴이 다를 가능성이 있으며, 이는 소주제 4에서 상세 분석한다.**

# %% [markdown]
# ### 3.7 기술통계 요약표

# %%
# 전체 기술통계
desc = df[["rating", "review_count", "price_level"]].describe().round(3)
desc.index = ["관측치 수", "평균", "표준편차", "최솟값", "Q1(25%)", "중앙값(50%)", "Q3(75%)", "최댓값"]
desc.columns = ["평점(rating)", "리뷰 수(review_count)", "가격대(price_level)"]
print("Table 1. 주요 변수 기술통계량")
print(desc.to_string())

# %% [markdown]
# ### 3.8 EDA 핵심 요약
# 
# | 발견 | 내용 | 시사점 |
# |------|------|--------|
# | 평점 분포 | 좌편향, 평균 4.14 | 온라인 평점의 긍정 편향 특성 |
# | 가격대-평점 | 상관 거의 없음 (r ≈ 0.01) | 비싼 곳이 평점 높지 않음 |
# | 리뷰 수-평점 | 약한 음의 상관 (r ≈ -0.10) | 리뷰 많을수록 평점 약간 하락 |
# | 행정구별 | 시각적 차이 존재 | 통계 검정 필요 |
# | 업종별 | 유사하나 교호작용 가능성 | 소주제 4에서 상세 분석 |

# %% [markdown]
# ---
# **다음 단계:** 소주제별 통계 분석 (Part 2에서 진행)